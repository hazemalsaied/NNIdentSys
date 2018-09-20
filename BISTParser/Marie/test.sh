
set -x


PROJH=$HOME/Documents/PROJETS/GRAPH_PARSER/git/marie-parsing
# on CLC
#PROJH=/home/mcandito/PROJETS/marie-parsing

DATAH=$HOME
# on CLC
#DATAH=/data/mcandito_bk

#D=$DATAH/Documents/PROJETS/DEEP_FRENCH/data/gold/surf
# avec POS prédites
D=$DATAH/Documents/PROJETS/DEEP_FRENCH/data/pred/mrf-ftb

W_EMB=$DATAH/Documents/ALPAGE/alparticles/alparticles/projetsLI1617/ressources/pretrained_embeddings/vecs100-linear-frwiki 
EVAL07=$HOME/Documents/OUTILS/evaldep/eval07.pl

timestamp=$(date "+%Y-%m-%d-%H-%M-%S")
LR=0.03
MAX_EPOCHS=8
#MAX_EPOCHS=2
OPTIM=Adagrad
#GRU_OR_LSTM=gru
GRU_OR_LSTM=lstm
W_EMBEDDING_DIM=100
P_EMBEDDING_DIM=8
ERROR_EXPLORATION_P=0.9
AGGRESSIVE_ERROR_EXPLORATION_P=0
FIRST_EPOCH_FOR_EXPL=1

TRAIN=surf.ftb.train.jk-p10-l3.predmorph.goldsynt.conll
#TRAIN=surf.ftb.test.predmorph.goldsynt.conll
DEV=surf.ftb.dev.predmorph.goldsynt.conll
#TRAIN=surf.ftb.train
#DEV=surf.ftb.dev
MODEL=surf.ftb.trainjk.model-labeled-wembNO-wemb$W_EMBEDDING_DIM-pemb$P_EMBEDDING_DIM-lr$LR-$OPTIM-$GRU_OR_LSTM-errorexpl$ERROR_EXPLORATION_P-$timestamp
#MODEL=surf.ftb.test.model-$timestamp

# # pour petits tests, sur sequoia
# D=$DATAH/Documents/PROJETS/DEEP_FRENCH/data/gold/surf
# #TRAIN=surf.sequoia.train
# TRAIN=surf.sequoia.test
# DEV=surf.sequoia.dev
# MODEL=$TRAIN.model-labeled-wembNO-lr$LR-$OPTIM-$GRU_OR_LSTM-errorexpl$ERROR_EXPLORATION_P-$timestamp

# # for debug
# D=$PROJH/corpus
# TRAIN=sequoia-first800.conll
# DEV=$TRAIN
# MODEL=$TRAIN.DEBUG.model_errorpropag0.5

# train using $DEV as validation set
# use -e for early stopping
#    Otherwise, even if loss increases, training will continue.
#    but the saved model will be that with minimum loss

# without pretrained embeddings
# to use pretrained embeddings : add -p $W_EMB
python $PROJH/src/test.py -w $W_EMBEDDING_DIM -c $P_EMBEDDING_DIM -x $ERROR_EXPLORATION_P -a $AGGRESSIVE_ERROR_EXPLORATION_P -y $FIRST_EPOCH_FOR_EXPL -o $OPTIM -r $LR -g $GRU_OR_LSTM -n $MAX_EPOCHS -l $MODEL.log -v $D/$DEV train $D/$TRAIN $MODEL  

# parse train
python $PROJH/src/test.py test $D/$TRAIN $MODEL > ${TRAIN%.goldsynt.conll}.parsed-$MODEL
# parse dev
python $PROJH/src/test.py test $D/$DEV $MODEL > ${DEV%.goldsynt.conll}.parsed-$MODEL

$EVAL07 -g $D/$TRAIN -s ${TRAIN%.goldsynt.conll}.parsed-$MODEL | head -3 | tee ${TRAIN%.goldsynt.conll}.parsed-$MODEL.eval07

$EVAL07 -g $D/$DEV -s ${DEV%.goldsynt.conll}.parsed-$MODEL | head -3 | tee ${DEV%.goldsynt.conll}.parsed-$MODEL.eval07

echo eval on TRAIN >> $MODEL.log
cat ${TRAIN%.goldsynt.conll}.parsed-$MODEL.eval07 >> $MODEL.log
echo eval on DEV >> $MODEL.log
cat ${DEV%.goldsynt.conll}.parsed-$MODEL.eval07 >> $MODEL.log



