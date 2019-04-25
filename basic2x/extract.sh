#!/usr/bin/env sh

TOOLS=/path/to/caffe/build/tools

if [ $# -lt 3 ]
then
    echo "Usage: ./extract.sh iter batchsize batchnum [gpu]"
    exit
fi

GLOG_logtostderr=1 $TOOLS/extract_txt ./save/basic2x_iter_$1.caffemodel ./extract.prototxt pool5 ./basic2x_fea.txt no $2 $3 2>&1 | tee log_extract
