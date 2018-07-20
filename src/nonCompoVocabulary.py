import sys
from random import uniform

from config import configuration
from corpus import getTokens
from reports import tabs, seperator, doubleSep
from wordEmbLoader import unk, empty, number


class Vocabulary:
    def __init__(self, corpus):
        global embConf, initConf
        embConf = configuration["model"]["embedding"]
        initConf = configuration["model"]["embedding"]["initialisation"]

        self.tokenFreqs, self.posFreqs = getFrequencyDics(corpus)
        self.tokenIndices = indexateDic(self.tokenFreqs)
        self.posIndices = indexateDic(self.posFreqs)
        if configuration["xp"]["verbose"] == 1:
            sys.stdout.write(str(self))
            self.verify(corpus)

    def __str__(self):
        res = seperator + tabs + 'Vocabulary' + doubleSep
        res += tabs + 'Tokens := {0} * POS : {1}'.format(len(self.tokenIndices), len(self.posIndices)) \
            if not configuration["xp"]["compo"] else ''
        res += seperator
        return res

    def getIndices(self, tokens, getPos=False, getToken=False):
        tokenTxt, posTxt = attachTokens(tokens)
        if tokenTxt in self.tokenIndices:
            tokenIdx = self.tokenIndices[tokenTxt]
        else:
            tokenIdx = self.tokenIndices[unk]
        if posTxt in self.posIndices:
            posIdx = self.posIndices[posTxt]
        else:
            posIdx = self.posIndices[unk]
        return tokenIdx, posIdx
    def getEmptyIdx(self, getPos=False, getToken=False):
        pass
    def verify(self, corpus):
        importTokens = 0
        for t in corpus.mweTokenDictionary:
            if t not in self.tokenIndices:
                importTokens += 1
        if importTokens:
            sys.stdout.write(tabs + 'Important words not in vocabulary {0}\n'.format(importTokens))
        importMWEs = 0
        for mwe in corpus.mweDictionary:
            mwe = mwe.replace(' ', '_')
            if mwe not in self.tokenIndices:
                importMWEs += 1
                print mwe
        if importMWEs:
            sys.stdout.write(tabs + 'MWE not in vocabulary {0}\n'.format(importMWEs))
        if unk not in self.tokenIndices or empty not in self.tokenIndices:
            sys.stdout.write(tabs + 'unk or empty is not in vocabulary\n')
        dashedKeys = 0
        for k in self.tokenIndices:
            if '_' in k:
                dashedKeys += 1
        # supposedDashedKeys = 0
        # for mwe in corpus.mweDictionary:
        #     supposedDashedKeys += len(mwe.split(' ')) - 1
        # sys.stdout.write(tabs + 'Suupposed dashed keys in vocabulary {0}\n'.format(supposedDashedKeys))
        sys.stdout.write(tabs + 'Dashed keys in vocabulary {0}\n'.format(dashedKeys))
        oneFreq = 0
        for k in self.tokenFreqs:
            if self.tokenFreqs[k] == 1:
                oneFreq += 1
        sys.stdout.write(tabs + 'One occurrence keys in vocabulary {0} / {1}\n'.
                         format(dashedKeys, len(self.tokenFreqs)))


def getFrequencyDics(corpus, freqTaux=1):
    tokenVocab, posVocab = {unk: freqTaux + 1, empty: freqTaux + 1}, {unk: freqTaux + 1, empty: freqTaux + 1}
    for sent in corpus.trainingSents:
        trans = sent.initialTransition
        while trans:
            if trans.configuration.stack:
                tokens = getTokens(trans.configuration.stack[-1])
                if tokens:
                    tokenTxt, posTxt = attachTokens(tokens)
                    for c in tokenTxt:
                        if c.isdigit():
                            tokenTxt = number
                    tokenVocab[tokenTxt] = 1 if tokenTxt not in tokenVocab else tokenVocab[tokenTxt] + 1
                    posVocab[posTxt] = 1 if posTxt not in posVocab else posVocab[posTxt] + 1
            trans = trans.next
    if embConf["compactVocab"]:
        sys.stdout.write(tabs + 'Compact Vocabulary cleaning:' + doubleSep)
        sys.stdout.write(tabs + 'Before : {0}\n'.format(len(tokenVocab)))
        for k in tokenVocab.keys():
            if k not in [empty, unk, number] and k.lower() not in corpus.mweTokenDictionary and '_' not in k:
                del tokenVocab[k]
        sys.stdout.write(tabs + 'After : {0}\n'.format(len(tokenVocab)))

    elif embConf["frequentTokens"]:
        sys.stdout.write(tabs + 'Non frequent word cleaning:' + doubleSep)
        sys.stdout.write(tabs + 'Before : {0}\n'.format(len(tokenVocab)))
        for k in tokenVocab.keys():
            if tokenVocab[k] <= freqTaux and '_' not in k and k.lower() not in corpus.mweTokenDictionary:
                if uniform(0, 1) < configuration["constants"]["alpha"]:
                    del tokenVocab[k]
        sys.stdout.write(tabs + 'After : {0}\n'.format(len(tokenVocab)))
    return tokenVocab, posVocab


def attachTokens(tokens):
    tokenTxt, posTxt = '', ''
    for t in tokens:
        tokenTxt += t.getTokenOrLemma() + '_'
        posTxt += t.posTag + '_'
    return tokenTxt[:-1], posTxt[:-1].lower()


def indexateDic(dic):
    res = dict()
    r = range(len(dic))
    for i, k in enumerate(dic):
        res[k] = r[i]
    return res
