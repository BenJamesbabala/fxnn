from __future__ import print_function

from sys import stdout
from random import randint
from functools import partial
import argparse as ap

import numpy as np
import scipy.io
import theano as th
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

import layers
from dataset import Cifar10Loader, MnistLoader
from optimize import VanillaSGD

th.config.floatX = 'float32'

g_parser = ap.ArgumentParser()
g_parser.add_argument('-m', '--model', default='fxnn', help='one of "fxnn", "cnn", "baseline"')
g_parser.add_argument('-d', '--dataset', default='MNIST', help='"MNIST" or "CIFAR10"')
g_parser.add_argument('-fsize', '--filter-size', type=int, default=3, help='filter size for "cnn" model, obsolete for FXNN')
g_parser.add_argument('-bsize', '--batch-size', type=int, default=64, help='self explanatory')
g_parser.add_argument('-pdpo', '--dropout-prob', type=float, default=0.25, help='probability of dropout during training, 0.0 to disable')
g_parser.add_argument('-nrec', '--num-records', type=int, default=100, help='number of frames recorded for data analysis')
g_parser.add_argument('-ipr', '--iters-per-record', type=int, default=100, help='SGD iterations per single record')
g_parser.add_argument('-lr', '--learn-rate', type=float, default=1e-3, help='SGD learn rate')
g_parser.add_argument('-ld', '--learn-decay', type=float, default=0.5, help='how much will learn rate decay at the end of training')
g_args = g_parser.parse_args()

#hyperparameters are in CAPS
MODEL = g_args.model; assert MODEL in ['fxnn', 'cnn', 'baseline']
DATASET = g_args.dataset; assert DATASET in ['MNIST', 'CIFAR10']
F_SIZE = g_args.filter_size; assert F_SIZE>0
B_SIZE = g_args.batch_size; assert B_SIZE>0
VALID_B_SIZE = 100 # batch size for validating, you should make size of validation set divisable by this
I_SIZE = {'MNIST':28,'CIFAR10':32}[DATASET] # FIXME: current GPU implementation does not support I_SIZE>512
TRAIN_PDPO = g_args.dropout_prob; assert 0.<=TRAIN_PDPO<=1.
LEARN_RATE = g_args.learn_rate
LEARN_DECAY = g_args.learn_decay #last_epoch_lr = first_epoch_lr*LEARN_DECAY, during training, learn rate decays exponentially
CNN_DIMS = [{'MNIST':1, 'CIFAR10':3}[DATASET],32,64]
MLP_DIMS = [384,192,10]
OP_ACT = T.nnet.relu#nonliner activation function

g_dataset = {'MNIST':MnistLoader, 'CIFAR10':Cifar10Loader}[DATASET]()
assert B_SIZE<=g_dataset.train_size
assert g_dataset.test_size%VALID_B_SIZE == 0
g_optimizer = VanillaSGD()
g_optimizer.lr = LEARN_RATE
g_params_di = {}

def model_baseline(s_x, s_pdpo_):
    '''very simple logistic regression model'''
    s_bsize = T.shape(s_x)[0]
    idim, odim = CNN_DIMS[0] * I_SIZE**2, MLP_DIMS[-1]
    return T.nnet.softmax(
        layers.op_dropout(layers.lyr_linear(
            'm',
            T.reshape(s_x, (s_bsize,idim)),
            idim, odim), s_pdpo_))

def model_cnn(s_x_, s_pdpo_, lyr_conv_=layers.lyr_conv):
    '''
    CNN model, can choose convolution type via argument "lyr_conv_"

    Builds a CNN of alternating conv/pool layers, followed by full connected layers

    Args:
        s_x_: input image batch, 4D tensor with shape NCHW (batch_size, channels, height, width)
        s_pdpo_: dropout probability, symbolic or constant scalar within range [0.0, 1.0]
        lyr_conv_: lyr_conv or lyr_simple_conv

    Returns: symbolic prediction from s_x_
    '''
    op_pool = partial(pool_2d, ds=(2,2), mode='max', ignore_border=True)
    s_bsize = T.shape(s_x)[0]

    s_features_li = [s_x]
    cnn_depth = len(CNN_DIMS)-1
    mlp_depth = len(MLP_DIMS)-1
    assert I_SIZE%(2**cnn_depth)==0, 'Convnet is too deep for image size, try changing depth or image size'
    for i, idim, odim in zip(range(cnn_depth),CNN_DIMS[:-1],CNN_DIMS[1:]):
        s_features_li.append(op_pool(
            layers.op_dropout(
                OP_ACT(lyr_conv_(
                    'conv%d'%i,
                    s_features_li[-1],
                    idim, odim)),
                s_pdpo_)))
    cnn_out_dims = CNN_DIMS[-1]*(I_SIZE//(2**cnn_depth))**2
    s_features_li.append(OP_ACT(
        layers.lyr_linear(
            'mlp0',
            T.reshape(s_features_li[-1], (s_bsize, cnn_out_dims)),
            cnn_out_dims,
            MLP_DIMS[0])))
    for i,idim,odim in zip(range(mlp_depth-1), MLP_DIMS[:-2], MLP_DIMS[1:-1]):
        s_features_li.append(OP_ACT(
            layers.lyr_linear(
                'mlp%d'%(i+1),
                s_features_li[-1],
                idim, odim)))
    return T.nnet.softmax(
        layers.lyr_linear(
            'mlp%d'%mlp_depth,
            s_features_li[-1], MLP_DIMS[-2], MLP_DIMS[-1]))

def build_model(model_):
    global s_x, s_y
    global fn_predict, fn_train, fn_record
    global g_optimizer
    global g_params_di

    layers.set_current_params(g_params_di)

    s_x = T.tensor4()
    s_y = T.ivector()
    s_pdpo = T.scalar()
    s_bsize = T.shape(s_x)[0]
    s_out = model_(s_x, s_pdpo)

    s_y_onehot = T.extra_ops.to_one_hot(s_y, MLP_DIMS[-1])
    s_loss = T.mean(-s_y_onehot*T.log(s_out + 1e-3))
    s_erate = T.mean(T.switch(T.eq(T.argmax(s_out, axis=1),T.argmax(s_y_onehot, axis=1)),0,1))

    no_dropout = [(s_pdpo, T.constant(0.,dtype=th.config.floatX))]
    fn_predict = th.function([s_x, s_y], {'pred':s_out, 'loss':s_loss, 'erate':s_erate}, givens=no_dropout)
    debug_fetches = {
        'x':s_x,
        'y':s_y,
    }
    debug_fetches.update(g_params_di)
    fn_record = th.function(
        [s_x, s_y], debug_fetches, givens=no_dropout
    )
    g_optimizer.compile(
        [s_x, s_y],
        s_loss,
        g_params_di.values(),
        fetches_={'loss':s_loss, 'erate':s_erate},
        givens_=[(s_pdpo, T.constant(TRAIN_PDPO, dtype=th.config.floatX))])
def train(niter_=100):
    global g_optimizer, g_tloss
    g_tloss = []
    g_terate = []
    try:
        for i in range(niter_):
            X,Y = g_dataset.load(
                np.random.randint(
                    0,g_dataset.train_size-1, B_SIZE, dtype='int32'))
            _ = g_optimizer.fit(X,Y)
            erate = _['erate']
            loss = _['loss']
            g_tloss.append(loss)
            g_terate.append(erate)
            print('Epoch %d: loss|error rate: %f|%f'%(i+1,loss,erate))
    except KeyboardInterrupt:
        print('User hit CTRL-C, abort')

def train_rec(fn_rec_, nipr_=100, nrec_=20):
    '''
    records numpy-arrays while training

    Args:
        fn_rec_: function returning dict of numpy arrays
        nipr_: iterations per record
        nrec_: total number of records
    '''
    global g_optimizer, fn_predict
    global g_tloss, g_terate, g_vloss, g_verate
    g_tloss = []
    g_terate = []
    g_vloss = []
    g_verate = []
    rec = {}
    def _validate():
        valid_loss, valid_erate = 0., 0.
        valid_size = g_dataset.test_size
        for batch in range(0,valid_size,VALID_B_SIZE):
            X, Y = g_dataset.load(np.arange(batch, batch+VALID_B_SIZE), is_test_=True)
            res = fn_predict(X,Y)
            valid_loss += res['loss']
            valid_erate += res['erate']
        valid_loss /= (valid_size/VALID_B_SIZE)
        valid_erate /= (valid_size/VALID_B_SIZE)
        return valid_loss, valid_erate

    def _train_iter():
        X,Y = g_dataset.load(
            np.random.randint(
                0,g_dataset.train_size-1, B_SIZE, dtype='int32'))
        res = g_optimizer.fit(X,Y)
        loss = res['loss']
        erate = res['erate']
        return loss, erate

    valid_loss, valid_erate = _validate()
    ndigits = len(str(nrec_*nipr_))
    print('\nIter '+'0'*ndigits+'/%d: loss|erate --------|------%% valid loss|erate %f|%6.2f%%'%(nipr_*nrec_,valid_loss,valid_erate*100))
    try:
        for i in range(nrec_):
            g_optimizer.lr = LEARN_RATE*(LEARN_DECAY**(i/(nrec_-1)))
            train_loss, train_erate = 0., 0.
            #do minibatch iterations
            for j in range(nipr_):
                loss, erate = _train_iter()
                train_loss += loss
                train_erate += erate
                g_tloss.append(loss)
                g_terate.append(erate)
                stdout.write('.')
                stdout.flush()
            train_loss /= nipr_
            train_erate /= nipr_
            #check test error rate
            valid_loss, valid_erate = _validate()

            g_vloss.append(valid_loss)
            g_verate.append(valid_erate)

            rec_frame = fn_rec_()
            for k in rec_frame:
                if k not in rec:
                    rec[k] = []
                rec[k].append(np.asarray(rec_frame[k]))
            print(('\nIter %0'+str(ndigits)+'d/%d: loss|erate %f|%6.2f%% valid loss|erate %f|%6.2f%%')%(
                (i+1)*nipr_,
                nipr_*nrec_,
                train_loss,
                train_erate*100,
                valid_loss,
                valid_erate*100))
    except KeyboardInterrupt:
        print('User hit CTRL-C, abort')

    for k in rec:
        rec[k] = np.asarray(rec[k])
    rec['tloss'] = np.asarray(g_tloss, dtype='float32')
    rec['terate'] = np.asarray(g_terate, dtype='float32')
    rec['vloss'] = np.asarray(g_vloss, dtype='float32')
    rec['verate'] = np.asarray(g_verate, dtype='float32')
    return rec

def record_cb():
    '''recording callback, used to record a "frame" of numpy-arrays'''
    X,Y = g_dataset.load(list(range(10)), is_test_=True)
    return fn_record(X,Y)

def main():
    global g_rec, g_args
    print('Using dataset "%s".'%DATASET)
    print('Building model "%s" ... '%MODEL, end='')
    stdout.flush()
    build_model(
        model_={
            'fxnn':partial(model_cnn, lyr_conv_=layers.lyr_simple_conv),
            'cnn':model_cnn,
            'baseline':model_baseline
        }[MODEL])
    print(' done.'); stdout.flush()
    g_rec = train_rec(fn_rec_=record_cb, nipr_=g_args.iters_per_record, nrec_=g_args.num_records)
    savefile = 'rec-%s-%s.mat'%(DATASET,MODEL)
    scipy.io.savemat(savefile, g_rec)
    print('Saved recording to file '+savefile)

if __name__!='__main__':
    raise RuntimeError('This file is not supposed to be imported as a module.')
main()
