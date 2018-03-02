import logging

import numpy as np

from corpus import Token, getTokens

mwtDictionary = None


class NNExtractor:
    def __init__(self, corpus):
        global mwtDictionary
        mwtDictionary = corpus.mwtDictionary
        self.historyDic, self.distanceDic, self.syntacticFeatDic, self.stackLengthDic, self.isMWComponentTDic = \
            dict(), dict(), dict(), dict(), dict()
        self.featDics = [self.stackLengthDic, self.isMWComponentTDic, self.syntacticFeatDic, self.distanceDic,
                         self.historyDic]
        for sent in corpus:
            idx = 0
            dicList = extractSent(sent, corpus)
            for dic in dicList:
                self.featDics[idx].update(dic)
                idx += 1
        self.featureNum = len(self.stackLengthDic) + len(self.isMWComponentTDic) + len(self.syntacticFeatDic) + \
                          len(self.distanceDic) + len(self.historyDic)
        logging.warn('Extracted feature number: {0}'.format(self.featureNum))

    def vectorize(self, trans):
        result, idx = np.zeros(self.featureNum, dtype='int32'), 0
        transFeatDics = extractTrans(trans)
        for transFeatDic in transFeatDics:
            for key in transFeatDic:
                if key in self.featDics[idx]:
                    featIdx = self.featDics[idx].keys().index(key)
                    result[featIdx] = 1 if featIdx < len(result) else 0
            idx += 1
        return result


def extractSent(sent, corpus):
    historyDic, distanceDic, syntacticFeatDic, stackLengthDic, isMWComponentTDic = \
        dict(), dict(), dict(), dict(), dict()
    localDicList = [stackLengthDic, isMWComponentTDic, syntacticFeatDic, distanceDic, historyDic]
    trans = sent.initialTransition
    while trans:
        idx = 0
        dics = extractTrans(trans)
        for dic in dics:
            localDicList[idx].update(dic)
            idx += 1
        trans = trans.next
    return [stackLengthDic, isMWComponentTDic, syntacticFeatDic, distanceDic, historyDic]


def extractTrans(trans):
    stackLengthDic, isMWTDic = dict(), dict()
    stackLengthDic['StackLengthIs' + str(len(trans.configuration.stack))] = True
    if trans.configuration.stack and isinstance(trans.configuration.stack[-1], Token):
        if trans.configuration.stack[-1].getLemma() in mwtDictionary:
            isMWTDic['S0isMWT'] = True
    syntacticFeatDic = extractSyntacticInfo(trans)
    distanceDic = extractDistanceInfo(trans)
    historyDic = extractHistoryDic(trans)
    dics = [stackLengthDic, isMWTDic, syntacticFeatDic, distanceDic, historyDic]
    return dics


def extractSyntacticInfo(trans):
    syntacticFeatDic = dict()
    config = trans.configuration
    if config.stack and isinstance(config.stack[-1], Token):
        stack0 = config.stack[-1]
        if int(stack0.dependencyParent) == -1 or int(stack0.dependencyParent) == 0 or \
                        stack0.dependencyLabel.strip() == '' or not config.buffer:
            return dict()
        for bElem in config.buffer:
            if bElem.dependencyParent == stack0.position:
                biIdx = config.buffer.index(bElem)
                syntacticFeatDic['hasRighDep_' + bElem.dependencyLabel] = True
                syntacticFeatDic['S0_hasRighDep_' + bElem.dependencyLabel] = True
                syntacticFeatDic[
                    'S0_B' + str(biIdx) + '_hasRighDep_' + bElem.dependencyLabel] = True
        if stack0.dependencyParent > stack0.position:
            for bElem in config.buffer[:5]:
                if bElem.position == stack0.dependencyParent:
                    biIdx = config.buffer.index(bElem)
                    syntacticFeatDic['S0_isGouvernedBy_' + str(biIdx)] = True
                    syntacticFeatDic['S0_isGouvernedBy_' + str(biIdx) + '_' + stack0.dependencyLabel] = True
                    break
        if len(config.stack) > 1:
            stack1 = config.stack[-2]
            if isinstance(stack1, Token):
                if stack0.dependencyParent == stack1.position:
                    syntacticFeatDic['SyntaxicRelation=' + stack0.dependencyLabel] = True
                elif stack0.position == stack1.dependencyParent:
                    syntacticFeatDic['SyntaxicRelation=' + stack1.dependencyLabel] = True
    return syntacticFeatDic


def extractDistanceInfo(trans):
    # Distance information
    config = trans.configuration
    sent = trans.sent
    distanceDic = dict()
    if config.stack:
        stackTokens = getTokens(config.stack[-1])
        if config.buffer:
            b0Idx = sent.tokens.index(config.buffer[0])
            s0Idx = sent.tokens.index(stackTokens[-1])
            distanceDic['S0B0Distance=' + str(b0Idx - s0Idx)] = True
        if len(config.stack) > 1 and isinstance(config.stack[-1], Token) \
                and isinstance(config.stack[-2], Token):
            s0Idx = sent.tokens.index(config.stack[-1])
            s1Idx = sent.tokens.index(config.stack[-2])
            distanceDic['S0S1Distance=' + str(s0Idx - s1Idx)] = True
    return distanceDic


def extractHistoryDic(trans):
    idx, history, historyDic = 0, '', dict()
    transition = trans.previous
    while transition and idx < 3:
        if transition.type:
            history += str(transition.type)
        transition = transition.previous
        idx += 1
    while len(history) <= 6:
        history += '-'
    historyDic['hisotry3=' + history] = True
    return historyDic
