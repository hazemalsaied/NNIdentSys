#!/bin/bash
sudo-g5k cp /home/halsaied/miniconda2/lib/libcudnn* /usr/local/lib/
cd /home/halsaied/NNIdenSys/
env MKL_THREADING_LAYER=GNU python src/xpNonCompo.py $xpLbl
