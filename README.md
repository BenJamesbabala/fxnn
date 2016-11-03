# fxnn - train CNNs faster and better using *fixed* convolution kernel

## What is this?

We found training CNN with a fixed *untrainable* 3x3 convolution followed by a *trainable* 1x1 convolution(technically just a GEMM) can *improve* CNN training convergence speed while requiring *less computation*. Currently this is supported with empirical evidence on small datasets including MNIST and CIFAR10. While the scalability of this method remains to be confirmed.

## A test run

Using our model:

    $ python3 fxnn.py --model=fxnn
    Using dataset "MNIST".
    Building model "fxnn" ...  done.
    Iter 00000/10000: loss|erate --------|------% valid loss|erate 0.455130| 87.15%
    Iter 00100/10000: loss|erate 0.125169| 39.05% valid loss|erate 0.053871| 16.81%
    Iter 00200/10000: loss|erate 0.060676| 19.86% valid loss|erate 0.037600| 11.66%
    Iter 00300/10000: loss|erate 0.046738| 14.86% valid loss|erate 0.032653| 10.47%
    Iter 00400/10000: loss|erate 0.040513| 13.14% valid loss|erate 0.029721|  9.54%
    ...
    Iter 00800/10000: loss|erate 0.030275|  9.95% valid loss|erate 0.022837|  7.29%
    ...
    Iter 01600/10000: loss|erate 0.022274|  7.19% valid loss|erate 0.017536|  5.54%
    Iter 01700/10000: loss|erate 0.024600|  7.86% valid loss|erate 0.017437|  5.56%
    ...
    Iter 03200/10000: loss|erate 0.017359|  5.42% valid loss|erate 0.013204|  4.18%
    Iter 03300/10000: loss|erate 0.015867|  5.00% valid loss|erate 0.012732|  4.20%
    ...
    Iter 06400/10000: loss|erate 0.013050|  4.33% valid loss|erate 0.009067|  3.00%
    Iter 06500/10000: loss|erate 0.011084|  3.47% valid loss|erate 0.009253|  3.02%
    ...
    Iter 09800/10000: loss|erate 0.011598|  3.64% valid loss|erate 0.007728|  2.55%
    Iter 09900/10000: loss|erate 0.012005|  3.77% valid loss|erate 0.007384|  2.39%
    Iter 10000/10000: loss|erate 0.011140|  3.53% valid loss|erate 0.007699|  2.58%


Using generic CNN model:

    $ python3 fxnn.py --model=cnn
    Using dataset "MNIST".
    Building model "cnn" ...  done.
    Iter 00000/10000: loss|erate --------|------% valid loss|erate 0.467279| 89.65%
    Iter 00100/10000: loss|erate 0.240944| 66.69% valid loss|erate 0.117343| 37.99%
    Iter 00200/10000: loss|erate 0.105306| 34.14% valid loss|erate 0.069288| 21.57%
    Iter 00300/10000: loss|erate 0.073431| 23.20% valid loss|erate 0.053945| 16.85%
    Iter 00400/10000: loss|erate 0.061541| 18.72% valid loss|erate 0.045496| 13.99%
    ...
    Iter 00800/10000: loss|erate 0.041054| 13.19% valid loss|erate 0.032550|  9.93%
    ...
    Iter 01600/10000: loss|erate 0.030469|  9.83% valid loss|erate 0.023477|  7.14%
    Iter 01700/10000: loss|erate 0.029861|  9.42% valid loss|erate 0.022976|  6.90%
    ...
    Iter 03200/10000: loss|erate 0.023401|  7.38% valid loss|erate 0.017682|  5.28%
    Iter 03300/10000: loss|erate 0.023761|  7.50% valid loss|erate 0.017295|  5.16%
    ...
    Iter 06400/10000: loss|erate 0.017577|  5.62% valid loss|erate 0.013102|  4.01%
    Iter 06500/10000: loss|erate 0.017500|  5.50% valid loss|erate 0.013369|  4.07%
    ...
    Iter 09800/10000: loss|erate 0.015557|  4.86% valid loss|erate 0.011510|  3.63%
    Iter 09900/10000: loss|erate 0.014504|  4.45% valid loss|erate 0.011420|  3.57%
    Iter 10000/10000: loss|erate 0.013800|  4.22% valid loss|erate 0.011232|  3.52%

Above experiment was run on same overall model architechture, with only difference being convolution layer.
Yeah I know that is not quite close to state of art, but the point is to compare and provide empirical evidence.


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

### Run experiment

To run experiment with default setting.

`python3 fxnn.py`

To view help on hyperparameters setting.

`python3 fxnn.py -h`

**NOTE:** CIFAR10 dataset tend to converge much slower than MNIST, you may want longer training like:

`python3 fxnn.py -d CIFAR10 -nrec 1000 -ipr 250`

### Analyse data

training/validation loss curve and model parameters during training will be recorded in `rec-$(dataset)-$(model).mat` file (MATLAB format). Open the file with your favourite tool to analyse.


## How it works

The modified convolution layer takes M image channels and generates 5M image channels via 5 fixed 3x3 convolutions. Then all channels were fed into trainable 1x1 convolution to generate N output channels.

`M --(3x3 fixed)-> 5M --(1x1)-> N`

3x3 convolution kernels are following ones:

     [1, 2, 1] [ 1, 2, 1] [ 1, 0,-1] [ 1, 0,-1] [ 0, 0, 0]
     [2, 4, 2] [ 0, 0, 0] [ 2, 0,-2] [ 0, 0, 0] [ 0, 1, 0]
     [1, 2, 1] [-1,-2,-1] [ 1, 0,-1] [-1,-0, 1] [ 0, 0, 0]

All the above convolution kernels are separable (Note two of them are SOBEL edge detectors). If implemented correctly, this can run much faster than state-of-art generic 3x3 convolution algorithm.
We've written a slightly optimized version of this operation with CUDA, in `kernels/simple_conv.cu`.

## Known Bugs/Issues

- Currently does not support theano `gpuarray` backend.
- GPU op implementation cannot take image larger than 512x512.

## Credits

Thanks to Richard Marko's `python-mnist` for doing MNIST preprocessing.

## References

TODO
