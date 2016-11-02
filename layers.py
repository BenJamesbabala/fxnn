from time import time
from math import sqrt, pi

from six.moves import cPickle as pickle

import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

_default_params = {}
_g_params_di = _default_params

def set_current_params(di_):
    global _g_params_di
    _g_params_di = di_

def save_params(f_):
    #FIXME: this does not save shared_variable properties like "strict" or "allow_downcast"
    global _g_params_di
    pickle.dump(f_, {k:v.get_value() for k,v in _g_params_di.items()})

def load_params(f_):
    global _g_params_di
    di = pickle.load(f_)
    for k,v in di.items():
        _g_params_di[k] = th.shared(v, name=k)
    pickle.dump(f_, {k:v.get_value() for k,v in _g_params_di.items()})

def _get_sharedv(name_, shape_, init_range_=None, dtype_=th.config.floatX):
    '''
    get a shared tensor variable with name_, return existing one if exists, otherwise create a new one
    behaves like tf.get_variable
    '''
    global _g_params_di
    if name_ in _g_params_di:
        #TODO: add shape/dtype check?
        return _g_params_di[name_]
    if init_range_ is None:
        v = th.shared(
            np.zeros(shape_,dtype=dtype_),
            name=name_
        )
    else:
        v = th.shared(
            np.asarray(np.random.uniform(
                *init_range_,
                size=shape_),
                dtype=dtype_),
            name=name_
        )
    _g_params_di[name_] = v
    return v

g_rng = RandomStreams(seed=int(time()*100)%(2**32))

def op_dropout(s_x_, s_p_):
    return s_x_ * g_rng.binomial(n=1, p=1.-s_p_, size=T.shape(s_x_), dtype=th.config.floatX)

if th.config.device[:3] == 'gpu':
    from op_simpleconv import simple_conv
    def lyr_simple_conv(name_, s_x_, idim_, odim_, init_scale_=None):
        global _g_params_di
        name_W = name_+'_w'
        name_B = name_+'_b'
        ir = 1.4/sqrt(idim_*5+odim_) if init_scale_ is None else init_scale_
        v_W = _get_sharedv(name_W, (idim_*5,odim_), (-ir,ir))
        v_B = _get_sharedv(name_B, (odim_,))
        s_y = simple_conv(s_x_)
        return (T.dot(s_y.transpose(0,2,3,1), v_W) + v_B).transpose(0,3,1,2)
else:
    def lyr_simple_conv(name_, s_x_, idim_, odim_, init_scale_=None):
        global _g_params_di
        name_W = name_+'_w'
        name_B = name_+'_b'
        ir = 1.4/sqrt(idim_*5+odim_) if init_scale_ is None else init_scale_
        v_W = _get_sharedv(name_W, (idim_*5,odim_), (-ir,ir))
        v_B = _get_sharedv(name_B, (odim_,))
        ker = np.asarray([
            [[[-1.,0.,1.]]],
            [[[1.,2.,1.]]]
        ], dtype=th.config.floatX
        )
        (2,1,3,1)
        ker2 = np.transpose(ker, (0,1,3,2))

        s_shape = T.shape(s_x_)
        s_x = T.reshape(s_x_, (s_shape[0]*s_shape[1], 1, s_shape[2], s_shape[3]))
        s_i1 = T.reshape(T.nnet.conv2d(s_x, ker, border_mode='half', filter_shape=(2,1,1,3)), (s_shape[0]*s_shape[1]*2, 1, s_shape[2], s_shape[3]))
        s_i2 = T.reshape(T.nnet.conv2d(s_i1, ker2, border_mode='half', filter_shape=(2,1,3,1)), (s_shape[0], s_shape[1]*4, s_shape[2], s_shape[3]))
        s_y = T.join(1, s_i2, s_x_)
        return (T.dot(s_y.transpose(0,2,3,1), v_W) + v_B).transpose(0,3,1,2)

def lyr_conv(name_, s_x_, idim_, odim_, fsize_=3, init_scale_ = None):
    global _g_params_di
    name_conv_W = '%s_w'%name_
    name_conv_B = '%s_b'%name_
    ir = 1.4/sqrt(idim_*fsize_*fsize_+odim_) if init_scale_ is None else init_scale_
    v_conv_W = _get_sharedv(name_conv_W, (odim_,idim_,fsize_,fsize_),(-ir,ir))
    v_conv_B = _get_sharedv(name_conv_B, (odim_))
    return T.nnet.conv2d(
        s_x_, v_conv_W,
        filter_shape=(odim_, idim_, fsize_, fsize_),
        border_mode = 'half'
    )+v_conv_B.dimshuffle('x',0,'x','x')

def lyr_linear(name_, s_x_, idim_, odim_, init_scale_=None):
    global _g_params_di
    name_W = name_+'_w'
    name_B = name_+'_b'
    ir = 1.4/sqrt(idim_+odim_) if init_scale_ is None else init_scale_
    v_W = _get_sharedv(name_W, (idim_,odim_), (-ir,ir))
    v_B = _get_sharedv(name_B, (odim_,))
    return T.dot(s_x_, v_W) + v_B

