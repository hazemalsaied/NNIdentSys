from corpus import getRelevantModelAndNormalizer
from modelLinear import getFeatures
from transitions import *


def parse(sents, model, vectorizer=None, linearModels=None, linearVecs=None, mlpModels=None):
    for sent in sents:
        sent.identifiedVMWEs, sent.initialTransition, sentEmbs, tokens = [], None, None, None
        if configuration['xp']['kiperwasser']:
            tokenIdxs, posIdxs = model.getIdxs(sent)
            sentEmbs = model.getContextualizedEmbs(tokenIdxs, posIdxs)
        elif configuration['xp']['kiperComp']:
            tokens = sent.tokens[:5]
            tokenIdxs, posIdxs = model.getIdxs(tokens)
            sentEmbs = model.getContextualizedEmbs(tokenIdxs, posIdxs)
        sent.initialTransition = Transition(None, isInitial=True, sent=sent)
        t = sent.initialTransition
        while not t.isTerminal():
            newT = nextTrans(t, sent, model, vectorizer, sentEmbs, tokens, linearModels=linearModels,
                             linearVecs=linearVecs, mlpModels=mlpModels)
            newT.apply(t, sent, parse=True, isClassified=newT.isClassified)
            t = newT
            if configuration['xp']['kiperComp']:
                sentEmbs, tokens = refreshSentEmb(t, tokens, model, sentEmbs)


def nextTrans(t, sent, model, vectorizer, sentEmbs=None, tokens=None, linearModels=None, linearVecs=None,
              mlpModels=None):
    legalTansDic, predictedTrans = t.getLegalTransDic(), []
    if len(legalTansDic) == 1:
        return initialize(legalTansDic.keys()[0], sent)
    if configuration['xp']['kiperwasser']:
        predictedTrans = model.predict(t, sentEmbs)
    elif configuration['xp']['kiperComp']:
        predictedTrans = model.predict(t, sentEmbs, tokens)
    elif configuration['xp']['rnn'] or configuration['xp']['rnnNonCompo']:
        probVector = model.predict(t)
        predictedTrans = sorted(range(len(probVector)), key=lambda k: probVector[k], reverse=True)
    elif configuration['xp']['linear']:
        featDic = getFeatures(t, sent)
        if mlpModels:
            mlpModel, mlpNormalizer = getRelevantModelAndNormalizer(sent, None, mlpModels, None, True)
            probVector = mlpModel.predict(t)
            predictedTrans = sorted(range(len(probVector)), key=lambda k: probVector[k], reverse=True)
            featDic['MLP_Prediction'] = predictedTrans[0]
        if configuration['others']['svmScikit']:
            predTransValue = model.predict(vectorizer.transform(featDic))[0]  # .toarray()
            predTrans = getType(predTransValue)
            if predTrans in legalTansDic:
                trans = legalTansDic[predTrans]
                trans.isClassified = True
                return trans
            for t in [TransitionType.SHIFT, TransitionType.MERGE, TransitionType.REDUCE, TransitionType.MARK_AS_OTH]:
                if t in legalTansDic:
                    trans = legalTansDic[t]
                    trans.isClassified = False
                    return trans
        elif configuration['others']['svm']:
            predictions = model.predict(featDic)
            for item in predictions:
                predTrans = getType(item[0])
                if predTrans in legalTansDic:
                    trans = legalTansDic[predTrans]
                    trans.isClassified = True
                    return trans
        else:
            probVector = model.decision_function(vectorizer.transform(featDic))[0]
            predictedTrans = sorted(range(len(probVector)), key=lambda k: probVector[k], reverse=True)
    else:
        probVector = model.predict(t, linearModels=linearModels, linearVecs=linearVecs)
        predictedTrans = sorted(range(len(probVector)), key=lambda k: probVector[k], reverse=True)
    for t in predictedTrans:
        transType = getType(t)
        if transType in legalTansDic:
            trans = legalTansDic[transType]
            trans.isClassified = True
            return trans


def refreshSentEmb(t, tokens, model, sentEmbs):
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
        tokenIdxs, posIdxs = model.getIdxs(tokens)
        sentEmbs = model.getContextualizedEmbs(tokenIdxs, posIdxs)
    return sentEmbs, tokens
