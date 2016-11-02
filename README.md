# fxnn - train CNNs faster and better using *fixed* convolution kernel

## What is this?

We found training CNN with a fixed untrainable 3x3 convolution followed by a trainable 1x1 convolution(technically just a GEMM) can *improve* CNN training convergence speed while requiring *less computation*. Currently this is supported with empirical evidence on small datasets including MNIST and CIFAR10. While the scalability of this method remains to be confirmed.

## A test run

Using our model:

    $ python3 fxnn.py --model=fxnn
    Using dataset "MNIST".
    Building model "fxnn" ...  done.
    Iter 00000/10000: loss|erate --------|------% valid loss|erate 0.483403| 92.03%
    Iter 00100/10000: loss|erate 0.124804| 38.81% valid loss|erate 0.049185| 15.48%
    Iter 00200/10000: loss|erate 0.058595| 19.19% valid loss|erate 0.038013| 12.35%
    Iter 00300/10000: loss|erate 0.045358| 14.77% valid loss|erate 0.030964|  9.88%

Using generic CNN model:

    $ python3 fxnn.py --model=cnn
    Using dataset "MNIST".
    Building model "cnn" ...  done.
    Iter 00000/10000: loss|erate --------|------% valid loss|erate 0.580004| 90.64%
    Iter 00100/10000: loss|erate 0.213693| 60.34% valid loss|erate 0.091578| 29.51%
    Iter 00200/10000: loss|erate 0.092413| 29.70% valid loss|erate 0.062275| 20.32%
    Iter 00300/10000: loss|erate 0.070973| 23.59% valid loss|erate 0.049436| 15.89%


## How to reproduce it:

### Clone this repository

`git clone https://github.com/khaotik/fxnn`

### Install required software

- python 3
- numpy/scipy
- theano
- PyCUDA (required if you wish to run on GPU)

Setup theano with GPU is recommanded.

### Download datasets

The code is written to run experiment with MNIST or CIFAR10 dataset. You should modify the code if you want to test on other datasets.

For MNIST, go to [here](http://yann.lecun.com/exdb/mnist/) and download all 4 .gz files, save them in `data/image/mnist/`. Uncompress them, then run `python3 mnist_pickler.py` in the same folder. MNIST dataset should be ready to go.

For CIFAR10, go to [here](https://www.cs.toronto.edu/~kriz/cifar.html) and download the python version, place it in `data/image/cifar10`. Just uncompress it and CIFAR10 dataset should be ready to go.

### Launch experiment

Just enter `python3 fxnn.py` to run experiment with default setting. `python3 fxnn.py -h` to view help on hyperparameters settting.

**NOTE:** CIFAR10 dataset tend to converge much slower than MNIST, you may want longer training like `python3 fxnn.py -d CIFAR10 -nrec 1000 -ipr 250`

### Analysis data

training/validation loss curve and model parameters during training will be recorded in `rec-$(dataset)-$(model).mat` file (MATLAB format) after training. Open the file with your favourite tool to analysis.


## How it works

The modified convolution layer takes M image channels and generates 5M image channels via 5 fixed 3x3 convolutions. Then all channels were fed into trainable 1x1 convolution to generate N output channels.

`N --(3x3 fixed)-> 5N --(1x1)-> M`

3x3 convolution kernels are following ones:

     [1, 2, 1] [ 1, 2, 1] [ 1, 0,-1] [ 1, 0,-1] [ 0, 0, 0]
     [2, 4, 2] [ 0, 0, 0] [ 2, 0,-2] [ 0, 0, 0] [ 0, 1, 0]
     [1, 2, 1] [-1,-2,-1] [ 1, 0,-1] [-1,-0, 1] [ 0, 0, 0]

All the above convolution kernels are separable (Note two of them are SOBEL edge detectors). If implemented correctly, this can run much faster than state-of-art generic 3x3 convolution algorithm.
We've written a slightly optimized version of this operation with CUDA, in `kernels/simple_conv.cu`.

## Known Bugs/Issues

- Currently does not support theano `gpuarray` backend.
- GPU op implementation cannot take image larger than 512x512

## Credits

Thanks to Richard Marko's `python-mnist` for doing MNIST preprocessing.

## References

TODO
xnn - train CNNs faster and better using fixed convolution kernel
