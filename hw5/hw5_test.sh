#!/bin/bash
wget https://www.dropbox.com/s/nr98jpqusjtvg0v/model_20191209_1.h5?dl=1 -O "model_20191209_1.h5"
wget https://www.dropbox.com/s/nl91rrjxu3v2rek/model_20191209_2.h5?dl=1 -O "model_20191209_2.h5"
wget https://www.dropbox.com/s/xyob2gleztmbjap/model_20191211.h5?dl=1 -O "model_20191211.h5"

python3 hw5_test.py $1 $2

