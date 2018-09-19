#!/bin/sh
# NNIdenSys/Results $sh evaluation.sh ST2 RNN 2018.September.14.
dataset=$1
xp=$2
date=$3
list="FR"
if [  $dataset == "ST2" ] || [  $dataset == "ST1" ] ||[  $dataset == "FTB" ]; then
if [  $xp == "RNN" ] || [  $xp == "MLP" ] || [  $xp == "Kiperwasser" ] || [  $xp == "Linear" ] ; then

	if [  $dataset == "ST2" ]; then
	    list="BG DE EL EN ES EU FA FR HE HI HR HU IT LT PL PT RO SL TR"
	else
	    if [  $dataset == "ST1" ]; then
	        list="BG CS DE EL ES FA FR HE HU ITLT MT PL PT RO SV SL TR"
	    fi
	fi
	for i in $list
        do
            echo "Evaluating:  $i"
            python3 bin/validate_cupt.py --input $dataset/$xp/$date$i.txt
            python3 bin/evaluate.py --gold Gold/$i/test.cupt --pred $dataset/$xp/$date$i.txt --train Gold/$i/train.dev.cupt
        done
else
	echo "You didn't choose correct xp mode!"
fi
else
	echo "You didnt choose correct dataset!"
fi
