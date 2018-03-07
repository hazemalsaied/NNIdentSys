#!/bin/bash
cd /home/halsaied/NNIdenSys/
time THEANO_FLAGS='floatX=float32,device=gpu'
env MKL_THREADING_LAYER=GNU python src/identification.py
