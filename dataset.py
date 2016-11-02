from six.moves import cPickle as pickle
from random import randint

import numpy as np
import theano as th

class Dataset(object):
    def __init__(self):
        self.train_size = 0
        self.test_size = 0
        self.labels = []
        self.datas = []
        pass
    def load(self, i_=None, is_test_=False, augment_=False):
        if is_test_:
            y_set = self.labels[self.train_size:]
            x_set = self.datas[self.train_size:]
        else:
            y_set = self.labels[:self.train_size]
            x_set = self.datas[:self.train_size]
        if i_ is None:
            i_ = randint(0, len(x_set)-1)
        elif type(i_) is int:
            i_ = [i_]
        if not augment_:
            return x_set[i_], y_set[i_]
        return type(self)._augment_data_batch(x_set[i_]),y_set[i_]

    def _augment_data_batch(v_img_):
        return v_img_

class Cifar10Loader(Dataset):
    def __init__(self):
        self.train_size = 50000
        self.test_size = 10000
        self.labels = np.empty(shape=(60000,), dtype=np.int32)
        self.datas = np.empty(shape=(60000,3,32,32), dtype=np.float32)
        for b in range(5):
            with open('./data/image/cifar10/data_batch_'+str(b+1), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                self.labels[10000*b:10000*(b+1)] = np.asarray(data[b'labels'], dtype=np.int32)
                self.datas[10000*b:10000*(b+1)]= np.reshape(np.asarray(data[b'data'], dtype=np.float32), (10000,3,32,32))/255.
        with open('./data/image/cifar10/test_batch', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.labels[50000:60000] = np.asarray(data[b'labels'], dtype=np.int32)
            self.datas[50000:60000]= np.reshape(np.asarray(data[b'data'], dtype=np.float32), (10000,3,32,32))/255.

    def _augment_data_batch(v_img_):
        for img in v_img_:
            if randint(0,65535)>32767:
                img[:,:,:] = img[:,:,::-1]
            gamma = np.random.uniform(0.77,1.3, size=(3,1,1))
            img **= gamma
            r,n = divmod(randint(0,31), 8)
            if r==0:
                img[:,:n,:] = 0.
            elif r==1:
                img[:,-n:, :] = 0.
            if r==2:
                img[:,:,:n] = 0.
            elif r==3:
                img[:,:,-n:] = 0.

class MnistLoader(Dataset):
    def __init__(self):
        self.train_size = 60000
        self.test_size = 10000
        self.labels = np.empty(shape=(70000,), dtype=np.int32)
        self.datas = np.empty(shape=(70000,1,28,28), dtype=np.float32)
        with open('./data/image/mnist/mnist-train.pkl', 'rb') as f:
            self.labels[:60000] = np.asarray(pickle.load(f), dtype=np.int32)
            self.datas[:60000] = np.expand_dims(np.asarray([pickle.load(f) for _ in range(60000)], dtype=th.config.floatX), 1)
        with open('./data/image/mnist/mnist-test.pkl', 'rb') as f:
            self.labels[60000:] = np.asarray(pickle.load(f), dtype=np.int32)
            self.datas[60000:] = np.expand_dims(np.asarray([pickle.load(f) for _ in range(10000)], dtype=th.config.floatX), 1)

    def _augment_data_batch(v_img_):
        return v_img_
