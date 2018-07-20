import modelKiperwasser
from transitions import *


def parse(corpus, clf, normalizer):
    printSentIdx, printSentNum = 0, 0
    initializeSent(corpus)
    for sent in corpus.testingSents:
        sentEmbs = None
        if configuration["xp"]["kiperwasser"]:
            tokenIdxs, posIdxs = modelKiperwasser.getIdxs(sent, normalizer)
            sentEmbs = clf.getContextualizedEmbs(tokenIdxs, posIdxs)
        sent.initialTransition = Transition(None, isInitial=True, sent=sent)
        transition = sent.initialTransition
        while not transition.isTerminal():
            newTransition = nextTrans(transition, sent, clf, normalizer, sentEmbs)
            newTransition.apply(transition, sent, parse=True)
            transition = newTransition
        if len(sent.vMWEs) > 1 and printSentIdx < printSentNum:
            print sent
            printSentIdx += 1


def nextTrans(transition, sent, clf, normalizer, sentEmbs=None):
    legalTansDic = transition.getLegalTransDic()
    if len(legalTansDic) == 1:
        return initialize(legalTansDic.keys()[0], sent)
    if configuration["xp"]["kiperwasser"]:
        transTypeValue = clf.predict(sentEmbs, transition)
    else:
        transTypeValue = clf.predict(transition, normalizer)
    if configuration["xp"]["pytorch"]:
        transTypeValue = transTypeValue.view(-1)
    # transTypeValue = normalizer.labelScaler.toRealLabels(transTypeValue)
    transType = getType(transTypeValue)
    if transType in legalTansDic:
        return legalTansDic[transType]
    if len(legalTansDic):
        return initialize(legalTansDic.keys()[0], sent)


def initializeSent(corpus):
    for sent in corpus.testingSents:
        sent.identifiedVMWEs = []
        sent.initialTransition = None
