#!/bin/bash
# Homogenizing and updating the PARSEME ST 1.0 French CONLLU files to UD v2.0
#
# Situation: the PARSEME ST 1.0 corpus in French is a merge of sequoia and UD1.4 corpora.
# Problem: the Sequoia corpus was not in UD format, so the tagsets and syntax trees are heterogeneous.
# New fact: since the release of the ST 1.0 corpora, a version of sequoia in UD2.0 became available.
# Goal: to upgrade both parts of the ST 1.0 CONLLU files, that is, sequoia and UD1.4
#
# 1) The original released PARSEME ST data is composed of (in this order):
# Test data: 
#    1667 first sentences of original sequoia
# Train data (17879 sentences): 
#    1432 remaining sentences of original sequoia
#  14553* sentences of UD v1.4 train +
#    1596 sentences of UD v1.4 dev +
#     298 sentences of UD v1.4 test

# * Actually, 14554 sentences, see remark below about sentence fr-ud-train_06073

# 2) Due to CONLL 2017 shared task, UD2.0 files have been split into train, test and dev as:
# Train data: 
#   14553 sentences of UD v1.4 train
# Dev data: 
#    1478 first sentences of UD v1.4 dev
# Test data (416 sentences):
#     118 remaining sentences of UD v1.4 dev +
#     298 sentences of UD v1.4 test

# 3) The sequoia UD2.0 conversion splits files randomly into train, dev and test.
# This makes it hard to align the released sequoia UD2.0 with the sentence order in original sequoia used in ST.
# Therefore, we used a full corpus version sent by email by Marie Candito.
# In this version, where the sentences are ordered in the original

# PATHS TO DATA:

# Sequoia converted to UD2.0 sent by email by Marie Candito
SEQUOIAUD2="$HOME/Work/corpora/sequoia/convert-UD2.0"

# UD 2.0 train and dev data downloaded from http://hdl.handle.net/11234/1-1983
UD2TRAIN="$HOME/Work/corpora/UD/ud-treebanks-v2.0/"

# UD 2.0 dev and test data downloaded from http://hdl.handle.net/11234/1-2184
UD2TEST="$HOME/Work/corpora/UD/ud-test-v2.0-conll2017/"

# PARSEME ST 1.0 release public git repo https://gitlab.com/parseme/sharedtask-data
PARSEME1="$HOME/Work/develop/parseme/sharedtask-data"

# PATHS TO TOOLS:

# New ST 'utilities' git repo from https://gitlab.com/parseme/utilities
UTILITIES="$HOME/Work/develop/parseme/utilities"

TMPFOLDER=./tmp-files

mkdir -p ${TMPFOLDER}

# 1) Retrieve the ST 1.0 parsemetsv files and concatenate them
# One sentence of the UD1.4 file is missing in UD2.0, remove it from parsemetsv

# Sentence fr-ud-train_08853 disappeared from UD2.0
cat ${PARSEME1}/FR/test.parsemetsv ${PARSEME1}/FR/train.parsemetsv | head -n 330957 > ${TMPFOLDER}/test-train-corrected.parsemetsv
cat ${PARSEME1}/FR/test.parsemetsv ${PARSEME1}/FR/train.parsemetsv | tail -n +331022 >> ${TMPFOLDER}/test-train-corrected.parsemetsv

# Sentence fr-ud-train_06073 had no sentid in ST 1.0 corpus (both .parsemetsv and .conllu). 
# This gave the impression that the number of sentences was the same in UD1.4 and UD2.0.
# However, UD2.0 has one sentence less (fr-ud-train_08853) than UD1.4.
# One sentence "more", to compensate for that, appeares in UD2.0 because the missing sent_id to fr-ud-train_06073 was corrected.

# 3) Retrieve Sequoia UD2.0 conversion

cp ${SEQUOIAUD2}/sequoia_ud_v2.0.conllu ${TMPFOLDER}

# 4) Retrieve UD 2.0 train and dev data. Dev contains only the first 1478 sentences of UD 1.4 dev

cp ${UD2TRAIN}/UD_French/fr-ud-train.conllu ${TMPFOLDER}/ud_v2.0_fr-train.conllu
cp ${UD2TRAIN}/UD_French/fr-ud-dev.conllu ${TMPFOLDER}/ud_v2.0_fr-dev.conllu

# 5) Retrieve UD 2.0 test data and separate into dev2 and train:
# Border between 1.4 dev and 1.4 test in UD2.0 test is line 8104, where UD1.4 test ends and UD1.4 dev remainder starts

head -n 8103 ${UD2TEST}/gold/conll17-ud-test-2017-05-09/fr.conllu > ${TMPFOLDER}/ud_v2.0_fr-test.conllu
tail -n +8104 ${UD2TEST}/gold/conll17-ud-test-2017-05-09/fr.conllu >> ${TMPFOLDER}/ud_v2.0_fr-dev.conllu

# 6) Concatenate all 2.0 conllu files in the order they appear in the join test-train.parsemetsv file

cat ${TMPFOLDER}/sequoia_ud_v2.0.conllu ${TMPFOLDER}/ud_v2.0_fr-train.conllu ${TMPFOLDER}/ud_v2.0_fr-dev.conllu  ${TMPFOLDER}/ud_v2.0_fr-test.conllu > data.conllu

# 7) Align both files in terms of tokens
${UTILITIES}/1.1/st-organizers/folia2parsemetsv.py --lang FR --input ${TMPFOLDER}/test-train-corrected.parsemetsv --conllu data.conllu > data.parsemetsv 2> alignment.log

# 8) Re-split data into train and test parsemetsv and conllu files, as in ST 1.0

${UTILITIES}/1.1/st-organizers/release-preparation/splitTrainTest.py --lang FR --input data.parsemetsv --conllu data.conllu 2> stats.md

# 9) Copy resulting files into current folder

mv OUT/* .
rm -r OUT
