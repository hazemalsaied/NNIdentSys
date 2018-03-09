import logging

import numpy as np

import v2featureSettings
from corpus import Token, getTokens

mwtDictionary = None
mweDictionary = None
mweTokenDictionary = None


class Extractor:
    def __init__(self, corpus):
        global mwtDictionary, mweDictionary, mweTokenDictionary
        mwtDictionary = corpus.mwtDictionary
        mweDictionary = corpus.mweDictionary
        mweTokenDictionary = corpus.mweTokenDictionary
        featSet = set()
        for sent in corpus:
            featSet.update(extractSent(sent))
        self.featList = list(featSet)
        self.featureNum = len(self.featList)
        logging.warn('Extracted feature number: {0}'.format(self.featureNum))

    def vectorize(self, trans):
        result = np.zeros(self.featureNum, dtype='int32')
        featSet = extractTrans(trans)
        for f in featSet:
            if f in self.featList:
                fIdx = self.featList.index(f)
                result[fIdx] = 1
        return result


def extractSent(sent):
    featSet = set()
    trans = sent.initialTransition
    while trans:
        featSet.update(extractTrans(trans))
        trans = trans.next
    return featSet


def extractTrans(trans):
    featSet = set()
    featSet.update(extractDicBased(trans))
    featSet.update(extractSyntaxic(trans))
    featSet.update(extractDistance(trans))
    featSet.update(extractHistory(trans))
    return featSet


def extractSyntaxic(trans):
    if not v2featureSettings.useSyntax:
        return set()
    featSet = set()
    config = trans.configuration
    if config.stack and isinstance(config.stack[-1], Token):
        stack0 = config.stack[-1]
        if int(stack0.dependencyParent) == -1 or int(stack0.dependencyParent) == 0 or \
                stack0.dependencyLabel.strip() == '' or not config.buffer:
            return dict()
        for bElem in config.buffer:
            if bElem.dependencyParent == stack0.position:
                biIdx = config.buffer.index(bElem)
                # syntacticFeatDic['hasRighDep_' + bElem.dependencyLabel] = True
                featSet.add('RighDep(S0)=' + bElem.dependencyLabel)
                featSet.add('RighDep(S0,B{0}) = {1}'.format(biIdx, bElem.dependencyLabel))
        if stack0.dependencyParent > stack0.position:
            for bElem in config.buffer[:5]:
                if bElem.position == stack0.dependencyParent:
                    biIdx = config.buffer.index(bElem)
                    featSet.add('Gouverner(S0) = B{0}'.format(biIdx))
                    featSet.add('Gouverner+Label(S0) = B{0} {1}'.format(biIdx, stack0.dependencyLabel))
                    break
        if len(config.stack) > 1:
            stack1 = config.stack[-2]
            if isinstance(stack1, Token):
                if stack0.dependencyParent == stack1.position:
                    featSet.add('SyntaxicRel(S0,S1)={0}'.format(stack0.dependencyLabel))
                elif stack0.position == stack1.dependencyParent:
                    featSet.add('SyntaxicRel(S0,S1)={0}'.format(stack1.dependencyLabel))
    return featSet


def extractDistance(trans):
    # Distance information
    config = trans.configuration
    sent = trans.sent
    featSet = set()
    if v2featureSettings.useStackLength:
        featSet.add('len(S)={0}'.format(len(trans.configuration.stack)))
    if config.stack:
        stackTokens = getTokens(config.stack[-1])
        if config.buffer:
            if v2featureSettings.useS0B0Distance:
                b0Idx = sent.tokens.index(config.buffer[0])
                s0Idx = sent.tokens.index(stackTokens[-1])
                featSet.add('Distance(S0,B0)={0}'.format(b0Idx - s0Idx))
        if len(config.stack) > 1 and isinstance(config.stack[-1], Token) \
                and isinstance(config.stack[-2], Token):
            if v2featureSettings.useS0S1Distance:
                s0Idx = sent.tokens.index(config.stack[-1])
                s1Idx = sent.tokens.index(config.stack[-2])
                featSet.add('Distance(S0,S1)={0}'.format(s0Idx - s1Idx))
    return featSet


def extractHistory(trans):
    idx, history, featSet = 0, '', set()
    transition = trans.previous
    while transition and idx < 3:
        if transition.type:
            history += str(transition.type.value)
        transition = transition.previous
        idx += 1
    while len(history) < 3:
        history += '-'
    if v2featureSettings.historyLength1:
        featSet.add('hisotry1={0}'.format(history[0]))
    if v2featureSettings.historyLength2:
        featSet.add('hisotry2={0}'.format(history[:2]))
    if v2featureSettings.historyLength3:
        featSet.add('hisotry3={0}'.format(history))
    return featSet


def extractDicBased(trans):
    config = trans.configuration
    featSet = set()
    if config.stack and isinstance(config.stack[-1], Token):
        if trans.configuration.stack[-1].getLemma() in mwtDictionary and v2featureSettings.smartMWTDetection:
            featSet.add('S0isMWT')
    if config.stack and (v2featureSettings.enhanceMerge or v2featureSettings.useLexicon):
        s0 = config.stack[-1]
        s0Tokens = getTokens(s0)
        s0LemmaStr = ''
        for t in s0Tokens:
            s0LemmaStr += t.getLemma() + ' '
            tIdx = s0Tokens.index(t)
            if t.getLemma() in mweTokenDictionary:
                featSet.add('S0_T{0}=MWE Token'.format(tIdx))
        s0LemmaStr = s0LemmaStr[:-1]
        if s0LemmaStr in mweDictionary:
            featSet.add('S0=MWE')
    if len(config.stack) > 1:
        s1 = config.stack[-2]
        s1Tokens = getTokens(s1)
        for t in s1Tokens:
            tIdx = s1Tokens.index(t)
            if t.getLemma() in mweTokenDictionary:
                featSet.add('S1_T{0}=MWE Token'.format(tIdx))
    return featSet
