#!/bin/bash
cd /home/halsaied/NNIdenSys/
time THEANO_FLAGS='floatX=float32,device=gpu'
python src/identification.py
