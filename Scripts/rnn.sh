#!/bin/bash
source /home/halsaied/miniconda2/bin/activate
cd /home/halsaied/NNIdenSys/
env MKL_THREADING_LAYER=GNU  python src/xpRnn.py
