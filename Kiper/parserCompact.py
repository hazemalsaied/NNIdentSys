import modelCompactKiper
from transitions import *


def parse(corpus, clf):
    for sent in corpus.testingSents:
        sent.identifiedVMWEs = []
        sent.initialTransition = None
        tokens = sent.tokens[:5]
        tokenIdxs, posIdxs = modelCompactKiper.getIdxs(tokens, clf)
        sentEmbs = clf.getContextualizedEmbs(tokenIdxs, posIdxs)
        sent.initialTransition = Transition(None, isInitial=True, sent=sent)
        t = sent.initialTransition
        while not t.isTerminal():
            newT = nextTrans(t, sent, clf, sentEmbs, tokens)
            newT.apply(t, sent, parse=True, isClassified=newT.isClassified)
            t = newT
            if t.configuration.buffer and t.configuration.buffer[-1] not in tokens \
                    and tokens[-1].position == t.configuration.buffer[0].position:
                existingTokens = []
                if t.configuration.stack:
                    for sItem in t.configuration.stack:
                        tokenBfr = getTokens(sItem)
                        for tItem in tokenBfr:
                            existingTokens.append(tItem)
                for iItem in t.configuration.buffer[:1]:
                    existingTokens.append(iItem)
                if len(existingTokens) == len(tokens):
                    if len(t.configuration.buffer) > 1:
                        tokens.append(t.configuration.buffer[1])
                else:
                    for i in range(len(tokens) - len(existingTokens)):
                        for tItem in tokens:
                            if tItem not in existingTokens:
                                tokens.remove(tItem)
                                break
                        if t.configuration.buffer and len(t.configuration.buffer) > i + 1:
                            tokens.append(t.configuration.buffer[i + 1])
                tokenIdxs, posIdxs = modelCompactKiper.getIdxs(tokens, clf)
                sentEmbs = clf.getContextualizedEmbs(tokenIdxs, posIdxs)


def nextTrans(t, sent, clf, sentEmbs, tokens):
    legalTansDic = t.getLegalTransDic()
    if len(legalTansDic) == 1:
        return initialize(legalTansDic.keys()[0], sent)
    predictedTrans = clf.predict(t, sentEmbs, tokens)
    for t in predictedTrans:
        transType = getType(t)
        if transType in legalTansDic:
            trans = legalTansDic[transType]
            trans.isClassified = True
            return trans
