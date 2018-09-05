import sys

import modelKiperwasser
from transitions import *

randomlySelectedTrans = 0


def parse(corpus, clf):
    for sent in corpus.testingSents:
        sent.identifiedVMWEs = []
        sent.initialTransition = None
        tokenIdxs, posIdxs = modelKiperwasser.getIdxs(sent, clf)
        sentEmbs = clf.getContextualizedEmbs(tokenIdxs, posIdxs)
        sent.initialTransition = Transition(None, isInitial=True, sent=sent)
        t = sent.initialTransition
        while not t.isTerminal():
            newT = nextTrans(t, sent, clf, sentEmbs)
            newT.apply(t, sent, parse=True, isClassified=newT.isClassified)
            t = newT
    global randomlySelectedTrans
    sys.stdout.write('\tRandomly Selected Transitions: {0}\n'.format(randomlySelectedTrans))


def nextTrans(t, sent, clf,  sentEmbs=None):
    legalTansDic = t.getLegalTransDic()
    if len(legalTansDic) == 1:
        return initialize(legalTansDic.keys()[0], sent)
    if configuration["xp"]["kiperwasser"]:
        predictedTrans = clf.predict(t, sentEmbs)
        if predictedTrans[0] > 1:
            print t
    for t in predictedTrans:
        transType = getType(t)
        if transType in legalTansDic:
            trans = legalTansDic[transType]
            trans.isClassified = True
            return trans
