import sys

import numpy as np

from config import configuration
from corpus import Token, getTokens
from reports import seperator, tabs, doubleSep

mwtDictionary = None
mweDictionary = None
mweTokenDictionary = None

featureSetting = None


class Extractor:
    def __init__(self, corpus):
        global featureSetting
        featureSetting = configuration['features']
        global mwtDictionary, mweDictionary, mweTokenDictionary
        mwtDictionary = corpus.mwtDictionary
        mweDictionary = corpus.mweDictionary
        mweTokenDictionary = corpus.mweTokenDictionary
        featSet = set()
        for sent in corpus:
            featSet.update(extractSent(sent))
        self.featList = list(featSet)
        self.featureNum = len(self.featList)
        if configuration['xp']['verbose'] == 1:
            sys.stdout.write(str(self))

    def __str__(self):
        report = seperator + tabs + 'Features' + doubleSep
        report += tabs + ' Features Num: {0}\n'.format(self.featureNum) + seperator
        return report

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
    featSet.update(extractDicBased(trans.configuration))
    featSet.update(abstractSyntaxic(trans.configuration))
    featSet.update(extractDistance(trans))
    featSet.update(extractHistory(trans))
    return featSet


def abstractSyntaxic(config):
    featSet = set()
    if not configuration['features']['syntax']:
        return featSet
    if config.stack:
        s0Tokens = getTokens(config.stack[-1])
        for t in reversed(s0Tokens):
            tIdx = len(s0Tokens) - s0Tokens.index(t) - 1
            if int(t.dependencyParent) == -1 or int(t.dependencyParent) == 0 or \
                    t.dependencyLabel.strip() == '' or not config.buffer:
                return dict()
            # Syntactic relation between S0 tokens
            for t1 in reversed(s0Tokens):
                if t1 is t:
                    continue
                t1Idx = len(s0Tokens) - s0Tokens.index(t1) - 1
                if t.dependencyParent == t1.position:
                    featSet.add('SyntaxicRel(S0T{0},S0T{1})={2}'.format(t1Idx, tIdx, t.dependencyLabel))
                elif t.position == t1.dependencyParent:
                    featSet.add('SyntaxicRel(S0T{0},S0T{1})={2}'.format(tIdx, t1Idx, t1.dependencyLabel))
                else:
                    featSet.add('SyntaxicRel(S0T{0},S0T{1})= null'.format(tIdx, t1Idx))
            # Righ dependence of S0 betwwen the first n elements of the buffer
            for bElem in config.buffer[:configuration['others']['bufferElements']]:
                if bElem.dependencyParent == t.position:
                    biIdx = config.buffer.index(bElem)
                    # syntacticFeatDic['hasRighDep_' + bElem.dependencyLabel] = True
                    featSet.add('Label(RightDependent(S0T{0}))={1}'.format(tIdx, bElem.dependencyLabel))
                    featSet.add('RightDependent(S0T{0}) = B{1}'.format(tIdx, biIdx))
            # S0 token Gouverner present in the buffer
            if t.dependencyParent > t.position:
                for bElem in config.buffer[:configuration['others']['bufferElements']]:
                    if bElem.position == t.dependencyParent:
                        biIdx = config.buffer.index(bElem)
                        featSet.add('Gouverner(S0T{0}) = B{1}'.format(tIdx, biIdx))
                        featSet.add('Label(Gouverner(S0T{0})) = {1}'.format(tIdx, t.dependencyLabel))
                        break
            if len(config.stack) > 1:
                s1Tokens = getTokens(config.stack[-2])
                # Syntactic relation between tokens of S0 and S1
                for tS1 in reversed(s1Tokens):
                    tS1Idx = len(s1Tokens) - s1Tokens.index(tS1) - 1
                    if t.dependencyParent == tS1.position:
                        featSet.add('SyntaxicRel(S1T{0},S0T{1})={2}'.format(tS1Idx, tIdx, t.dependencyLabel))
                    elif t.position == tS1.dependencyParent:
                        featSet.add('SyntaxicRel(S0T{0},S1T{1})={2}'.format(tIdx, tS1Idx, tS1.dependencyLabel))
                    else:
                        featSet.add('SyntaxicRel(S0T{0},S1T{1})= null'.format(tIdx, tS1Idx))
    # Syntactic relation between S1 tokens
    if config.stack and len(config.stack) > 1:
        s1Tokens = getTokens(config.stack[-2])
        for t in s1Tokens:
            for t1 in s1Tokens:
                if t is not t1:
                    tIdx = len(s1Tokens) - s1Tokens.index(t) - 1
                    t1Idx = len(s1Tokens) - s1Tokens.index(t1) - 1
                    if t.dependencyParent == t1.position:
                        featSet.add('SyntaxicRel(S1T{0},S1T{1})={2}'.format(t1Idx, tIdx, t.dependencyLabel))
                    elif t.position == tS1.dependencyParent:
                        featSet.add('SyntaxicRel(S1T{0},S1T{1})={2}'.format(tIdx, t1Idx, tS1.dependencyLabel))
                    else:
                        featSet.add('SyntaxicRel(S1T{0},S1T{1})= null'.format(tIdx, t1Idx))
    return featSet


def extractDistance(trans):
    # Distance information
    config = trans.configuration
    sent = trans.sent
    featSet = set()
    if configuration['features']['stackLength']:
        featSet.add('len(S)={0}'.format(len(trans.configuration.stack)))
    if config.stack:
        stackTokens = getTokens(config.stack[-1])
        if config.buffer:
            if configuration['features']['distanceS0b0']:
                b0Idx = sent.tokens.index(config.buffer[0])
                s0Idx = sent.tokens.index(stackTokens[-1])
                featSet.add('Distance(S0,B0)={0}'.format(b0Idx - s0Idx))
        if len(config.stack) > 1:
            s0Tokens = getTokens(config.stack[-1])
            s1Tokens = getTokens(config.stack[-1])
            if configuration['features']['distanceS0s1']:
                s0Idx = sent.tokens.index(s0Tokens[0])
                s1Idx = sent.tokens.index(s1Tokens[-1])
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
    if configuration['features']['history1']:
        featSet.add('hisotry1={0}'.format(history[0]))
    if configuration['features']['history2']:
        featSet.add('hisotry2={0}'.format(history[1]))
    if configuration['features']['history3']:
        featSet.add('hisotry3={0}'.format(history[2]))
    return featSet


def extractDicBased(config):
    featSet = set()
    if config.stack and isinstance(config.stack[-1], Token):
        if config.stack[-1].getLemma() in mwtDictionary :
            featSet.add('S0isMWT')
    if config.stack and configuration['features']['s0TokenIsMWEToken']:
        s0 = config.stack[-1]
        s0Tokens = getTokens(s0)
        s0LemmaStr = ''
        for t in s0Tokens:
            s0LemmaStr += t.getLemma() + ' '
            tIdx = s0Tokens.index(t)
            if t.getLemma() in mweTokenDictionary:
                featSet.add('S0_T{0}=MWE Token'.format(tIdx))
        s0LemmaStr = s0LemmaStr[:-1]
        if s0LemmaStr in mweDictionary and configuration['features']['s0TokensAreMWE']:
            featSet.add('S0=MWE')
        if config.stack and len(config.stack) > 1:
            s1Tokens = getTokens(config.stack[-2])
            for t in s1Tokens:
                tIdx = s1Tokens.index(t)
                if t.getLemma() in mweTokenDictionary:
                    featSet.add('S1_T{0}=MWE Token'.format(tIdx))
    if len(config.stack) > 1:
        s1 = config.stack[-2]
        s1Tokens = getTokens(s1)
        for t in s1Tokens:
            tIdx = s1Tokens.index(t)
            if t.getLemma() in mweTokenDictionary:
                featSet.add('S1_T{0}=MWE Token'.format(tIdx))
    return featSet
