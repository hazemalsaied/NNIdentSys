#!/bin/bash
source /home/halsaied/venv/bin/activate
cd /home/halsaied/NNIdenSys/
pwd
#echo "test identification.py CPU"
time THEANO_FLAGS='floatX=float32,device=gpu'
python src/identification.py
