# import modelKiperwasser
from transitions import *


def parse(corpus, clf, normalizer):
    for sent in corpus.testingSents:
        sent.identifiedVMWEs = []
        sent.initialTransition = None
        sentEmbs = None
        # if configuration["xp"]["kiperwasser"]:
        # tokenIdxs, posIdxs = modelKiperwasser.getIdxs(sent, clf)
        # sentEmbs = clf.getContextualizedEmbs(tokenIdxs, posIdxs)
        sent.initialTransition = Transition(None, isInitial=True, sent=sent)
        t = sent.initialTransition
        while not t.isTerminal():
            newT = nextTrans(t, sent, clf, normalizer, sentEmbs)
            newT.apply(t, sent, parse=True, isClassified=newT.isClassified)
            t = newT


def nextTrans(t, sent, clf, normalizer, sentEmbs=None):
    legalTansDic, predictedTrans = t.getLegalTransDic(), []
    if len(legalTansDic) == 1:
        return initialize(legalTansDic.keys()[0], sent)
    if configuration["xp"]["kiperwasser"]:
        predictedTrans = clf.predict(t, sentEmbs)
        if predictedTrans[0] > 1:
            print t
    elif configuration["xp"]["rnn"] or configuration["xp"]["rnnNonCompo"]:
        probVector = clf.predict(t)
        predictedTrans = sorted(range(len(probVector)), key=lambda k: probVector[k], reverse=True)
    # elif configuration["xp"]["linear"]:
    #     featDic = getFeatures(t, sent)
    #     if configuration['linear']['svm']:
    #         predTransValue = clf.predict(normalizer.transform(featDic).toarray())[0]
    #         predTrans = getType(predTransValue)
    #         if predTrans in legalTansDic:
    #             trans = legalTansDic[predTrans]
    #             trans.isClassified = True
    #             return trans
    #         for t in [TransitionType.SHIFT, TransitionType.MERGE, TransitionType.REDUCE, TransitionType.MARK_AS_OTH]:
    #             if t in legalTansDic:
    #                 trans = legalTansDic[t]
    #                 trans.isClassified = False
    #                 return trans
    #     else:
    #         probVector = clf.decision_function(normalizer.transform(featDic))[0]
    #         predictedTrans = sorted(range(len(probVector)), key=lambda k: probVector[k], reverse=True)
    else:
        probVector = clf.predict(t, normalizer)
        predictedTrans = sorted(range(len(probVector)), key=lambda k: probVector[k], reverse=True)
    for t in predictedTrans:
        transType = getType(t)
        if transType in legalTansDic:
            trans = legalTansDic[transType]
            trans.isClassified = True
            return trans
