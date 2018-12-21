#!/bin/sh
# NNIdenSys/Results $sh evaluation.sh ST2 RNN 2018.September.14.
dataset=ST2 #$1
xp=MLP #$2
date=12.11 #$3
eval=FixedSize
evalMini=fixedsize
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
            python3 bin/validate_cupt.py --input Output/$dataset/$xp/$eval/$date.$evalMini.$i.cupt
            python3 bin/evaluate.py --gold Output/$dataset/$xp/$eval/$date.$evalMini.$i.gold.cupt --pred Output/$dataset/$xp/$eval/$date.$evalMini.$i.cupt --train Output/$dataset/$xp/$eval/$date.$evalMini.$i.train.cupt
        done
else
	echo "You didn't choose correct xp mode!"
fi
else
	echo "You didnt choose correct dataset!"
fi
