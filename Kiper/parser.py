import modelKiperwasser
from transitions import *


def parse(sents, clf):
    for sent in sents:
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


def nextTrans(t, sent, clf, sentEmbs=None):
    legalTansDic = t.getLegalTransDic()
    if len(legalTansDic) == 1:
        return initialize(legalTansDic.keys()[0], sent)
    if configuration["xp"]["kiperwasser"]:
        predictedTrans = clf.predict(t, sentEmbs)
    for t in predictedTrans:
        transType = getType(t)
        if transType in legalTansDic:
            trans = legalTansDic[transType]
            trans.isClassified = True
            return trans
