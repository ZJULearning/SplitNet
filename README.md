# SplitNet
This is the repository for [SplitNet](http://) paper.

## Prerequisites
0. [Caffe](http://caffe.berkeleyvision.org/installation.html#prequequisites)

## Installation

    $ cd caffe
    $ "Adjust Makefile.config"
    $ make -j
    
The caffe contained in this repository is modified from [caffe-1.0](https://github.com/BVLC/caffe/releases/tag/1.0), including
- CornerCropLayer, for "image50" and "image56" setting
- LocalLayer, for "SplitNet-conv52" setting
- linear lr_policy
- extract_txt tool, for saving features to txt file

## Running Experiments
There are 5 file in each setting dir:
- train_test.prototxt: The network definition for training. You need modify /path/to/your/traininglmdb and /path/to/your/testinglmdb before using.
- solver.prototxt: The solver definition for training.
- extract.prototxt: The prototxt for extracting feature. You need modify /path/to/your/evaluationlmdb before using.
- train.sh: The script for training models.
- extract.sh: The script for extracting features.

You can train by simply running
    
    $ cd basic ( or basic2x or image50 or image56 or SplitNet-conv22 or SplitNet-conv52)
    $ ./train.sh

Once the training is completed, you can generate features by simply running

    $ cd basic ( or basic2x or image50 or image56 or SplitNet-conv22 or SplitNet-conv52)
    $ ./extract.sh
