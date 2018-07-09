import os
import sys
from random import uniform

import numpy as np
import word2vec
from keras.utils import to_categorical

from config import configuration
from corpus import getTokens
from reports import tabs, seperator, doubleSep
from wordEmbLoader import getRandomVector, getFreqDic, initialisePOS, unk, empty, number


class Vocabulary:
    def __init__(self, corpus):
        global embConf, initConf
        embConf = configuration["model"]["embedding"]
        initConf = configuration["model"]["embedding"]["initialisation"]

        if not configuration["xp"]["compo"]:
            self.attachedTokens, self.attachedPos = self.getAttachedVoc(corpus)
            self.posIndices, self.posEmbeddings = getPOSEmbeddingMatrices(corpus, self.attachedPos)
            self.tokenIndices, self.tokenEmbeddings = getTokenEmbeddingMatrices(corpus, self.attachedTokens)
        else:
            self.posIndices, self.posEmbeddings = getPOSEmbeddingMatrices(corpus)
            self.tokenIndices, self.tokenEmbeddings = getTokenEmbeddingMatrices(corpus)
            self.indices, self.embeddings = self.getEmbeddingMatrices(corpus)
            self.embDim = embConf["tokenEmb"] + embConf["posEmb"] if embConf["concatenation"] or embConf["usePos"] else \
                embConf["tokenEmb"]

        if configuration["xp"]["verbose"] == 1:
            sys.stdout.write(str(self))

    def __str__(self):
        res = seperator + tabs + 'Vocabulary' + doubleSep
        res += tabs + 'Tokens := {0} * POS : {1}'.format(len(self.attachedTokens), len(self.attachedPos)) \
            if not configuration["xp"]["compo"] else ''
        res += seperator
        return res

    def getEmbeddingMatrices(self, corpus):
        shouldInit = initConf["active"]
        if not embConf["concatenation"]:
            return None, None
        indices, embeddings, idx = self.generateUnknownKeys()
        for sent in corpus.trainingSents + corpus.testingSents:
            for token in sent.tokens:
                tokenKey = token.getTokenOrLemma() if token.getTokenOrLemma() in self.tokenIndices else unk
                key = tokenKey
                if embConf["usePos"]:
                    posKey = token.posTag.lower() if token.posTag.lower() in self.posIndices else unk
                    key += '_' + posKey
                if key not in indices:
                    if shouldInit:
                        if embConf["usePos"]:
                            embeddings[key] = np.concatenate((self.tokenEmbeddings[tokenKey],
                                                              self.posEmbeddings[posKey]))
                        else:
                            embeddings[key] = self.tokenEmbeddings[tokenKey]

                    indices[key] = idx
                    idx += 1
        return indices, embeddings

    def generateUnknownKeys(self):
        shouldInit = initConf["active"]
        indices, embeddings, idx = dict(), dict(), 0
        if embConf["usePos"]:
            for posTag in self.posIndices.keys():
                key = unk + '_' + posTag.lower()
                if key not in indices:
                    if shouldInit:
                        embeddings[key] = np.concatenate(
                            (self.tokenEmbeddings[unk], self.posEmbeddings[posTag.lower()]))
                    indices[key] = idx
                    idx += 1
                key1 = number + '_' + posTag.lower()
                if key1 not in indices:
                    if shouldInit:
                        embeddings[key1] = np.concatenate(
                            (self.tokenEmbeddings[number], self.posEmbeddings[posTag.lower()]))
                    indices[key1] = idx
                    idx += 1
        key = unk
        if embConf["usePos"]:
            key += '_' + unk
        if key not in indices:
            if shouldInit:
                if embConf["usePos"]:
                    embeddings[key] = np.concatenate((self.tokenEmbeddings[unk], self.posEmbeddings[unk]))
                else:
                    embeddings[key] = self.tokenEmbeddings[unk]
            indices[key] = idx
            idx += 1
        key1 = number
        if embConf["usePos"]:
            key1 += '_' + unk
        if key1 not in indices:
            if shouldInit:
                if initConf["pos"]:
                    embeddings[key1] = np.concatenate((self.tokenEmbeddings[number], self.posEmbeddings[unk]))
                else:
                    embeddings[key1] = self.tokenEmbeddings[unk]
            indices[key1] = idx
            idx += 1
        if shouldInit:
            if embConf["usePos"]:
                embeddings[empty] = np.zeros(
                    (len(self.tokenEmbeddings.values()[0]) + len(self.posEmbeddings.values()[0])))
            else:
                embeddings[empty] = np.zeros((len(self.tokenEmbeddings.values()[0])))
        indices[empty] = idx
        idx += 1
        return indices, embeddings, idx

    # def generateUnknownEmbedding(self):
    #     indices, embeddings, idx = dict(), dict(), 0
    #     if embConf["usePos"]:
    #         for posTag in self.posIndices.keys():
    #             key = unk + '_' + posTag.lower()
    #             if key not in indices:
    #                 embeddings[key] = np.concatenate(
    #                     (self.tokenEmbeddings[unk], self.posEmbeddings[posTag.lower()]))
    #                 indices[key] = idx
    #                 idx += 1
    #             key1 = number + '_' + posTag.lower()
    #             if key1 not in indices:
    #                 embeddings[key1] = np.concatenate(
    #                     (self.tokenEmbeddings[number], self.posEmbeddings[posTag.lower()]))
    #                 indices[key1] = idx
    #                 idx += 1
    #     key = unk
    #     if embConf["usePos"]:
    #         key += '_' + unk
    #     if key not in indices:
    #         if embConf["usePos"]:
    #             embeddings[key] = np.concatenate((self.tokenEmbeddings[unk], self.posEmbeddings[unk]))
    #         else:
    #             embeddings[key] = self.tokenEmbeddings[unk]
    #         indices[key] = idx
    #         idx += 1
    #     key1 = number
    #     if embConf["usePos"]:
    #         key1 += '_' + unk
    #     if key1 not in indices:
    #         if initConf["pos"]:
    #             embeddings[key1] = np.concatenate((self.tokenEmbeddings[number], self.posEmbeddings[unk]))
    #         else:
    #             embeddings[key1] = self.tokenEmbeddings[unk]
    #         indices[key1] = idx
    #         idx += 1
    #     if embConf["usePos"]:
    #         embeddings[empty] = np.zeros((len(self.tokenEmbeddings.values()[0]) + \
    # len(self.posEmbeddings.values()[0])))
    #     else:
    #         embeddings[empty] = np.zeros((len(self.tokenEmbeddings.values()[0])))
    #     indices[empty] = idx
    #     idx += 1
    #     return indices, embeddings, idx

    def generateUnknownIndices(self):
        indices, idx = dict(), 0
        if embConf["usePos"]:
            for posTag in self.posIndices.keys():
                key = unk + '_' + posTag.lower()
                if key not in indices:
                    indices[key] = idx
                    idx += 1
                key1 = number + '_' + posTag.lower()
                if key1 not in indices:
                    indices[key1] = idx
                    idx += 1
        key = unk
        if embConf["usePos"]:
            key += '_' + unk
        if key not in indices:
            indices[key] = idx
            idx += 1
        key1 = number
        if embConf["usePos"]:
            key1 += '_' + unk
        if key1 not in indices:
            indices[key1] = idx
            idx += 1
        indices[empty] = idx
        idx += 1
        return indices, idx

    def getIndices(self, tokens, getPos=False, getToken=False):
        result = []
        for token in tokens:
            key = self.getKey(token, getPos=getPos, getToken=getToken)
            if getPos:
                result.append(self.posIndices[key])
            elif getToken:
                result.append(self.tokenIndices[key])
            else:
                result.append(self.indices[key])
        return np.asarray(result)

    def getAttachedVoc(self, corpus):
        tokenVocab, posVocab = dict(), dict()
        for sent in corpus:
            trans = sent.initialTransition
            for token in sent.tokens:
                tokenTxt = token.getTokenOrLemma()
                if tokenTxt not in tokenVocab:
                    tokenVocab[tokenTxt] = 1
                else:
                    tokenVocab[tokenTxt] += 1
                posTxt = token.posTag.lower()
                if posTxt not in posVocab:
                    posVocab[posTxt] = 1
                else:
                    posVocab[posTxt] += 1
            while trans:
                if trans.configuration.stack:
                    tokens = getTokens(trans.configuration.stack[-1])
                    if tokens and len(tokens) > 1:
                        tokenTxt, posTxt = self.attachTokens(tokens)
                        if tokenTxt not in tokenVocab:
                            tokenVocab[tokenTxt] = 1
                        else:
                            tokenVocab[tokenTxt] += 1
                        if posTxt not in posVocab:
                            posVocab[posTxt] = 1
                        else:
                            posVocab[posTxt] += 1
                    if len(trans.configuration.stack) > 1:
                        tokens = getTokens(trans.configuration.stack[-2])
                        if len(tokens) > 1:
                            tokenTxt, posTxt = self.attachTokens(tokens)
                            if tokenTxt not in tokenVocab:
                                tokenVocab[tokenTxt] = 1
                            else:
                                tokenVocab[tokenTxt] += 1
                            if posTxt not in posVocab:
                                posVocab[posTxt] = 1
                            else:
                                posVocab[posTxt] += 1
                trans = trans.next
        if embConf["compactVocab"]:
            for k in tokenVocab.keys():
                if k.lower() not in corpus.mweTokenDictionary and '_' not in k:
                    del tokenVocab[k]

        if embConf["frequentTokens"]:
            for k in tokenVocab.keys():
                if tokenVocab[k] == 1 and k.lower() not in corpus.mweTokenDictionary and '_' not in k:
                    del tokenVocab[k]
            for k in posVocab.keys():
                if posVocab[k] == 1 and '_' not in k:
                    del posVocab[k]



        tokenVocab[unk] = 1
        tokenVocab[number] = 1
        tokenVocab[empty] = 1
        posVocab[unk] = 1
        posVocab[number] = 1
        posVocab[empty] = 1

        idx = 0
        for k in tokenVocab.keys():
            tokenVocab[k] = idx
            idx += 1
        idx = 0
        for k in posVocab.keys():
            posVocab[k] = idx
            idx += 1
        return tokenVocab, posVocab

    def getAttachedIndices(self, tokens):
        tokenTxt, posTxt = self.attachTokens(tokens)
        if tokenTxt in self.attachedTokens:
            tokenIdx = self.attachedTokens[tokenTxt]
        else:
            tokenIdx = self.attachedTokens[unk]
        if posTxt in self.attachedPos:
            posIdx = self.attachedPos[posTxt]
        else:
            posIdx = self.attachedPos[unk]
        return tokenIdx, posIdx

    def attachTokens(self, tokens):
        tokenTxt, posTxt = '', ''
        for t in tokens:
            tokenTxt += t.getTokenOrLemma() + '_'
            posTxt += t.posTag + '_'
        tokenTxt = tokenTxt[:-1]
        posTxt = posTxt[:-1].lower()
        return tokenTxt, posTxt

    def getKey(self, token, getPos=False, getToken=False):

        if getToken:
            if any(ch.isdigit() for ch in token.getTokenOrLemma()):
                key = number
                return key
            key = token.getStandardKey(getPos=False, getToken=True)
            if key in self.tokenIndices:
                return key
            return unk
        elif getPos:
            key = token.getStandardKey(getPos=True, getToken=False)
            if key in self.posIndices:
                return key
            return unk

        else:
            if any(ch.isdigit() for ch in token.getTokenOrLemma()):
                key = number
                posKey = token.getStandardKey(getPos=True, getToken=False)
                if key + '_' + posKey in self.indices:
                    return key + '_' + posKey
                return key + '_' + unk
            key = token.getStandardKey(getPos=False, getToken=False)
            if key in self.indices:
                return key
            return unk + '_' + unk

    def getEmptyIdx(self, getPos=False, getToken=False):
        if getPos:
            return self.posIndices[empty]
        if getToken:
            return self.tokenIndices[empty]
        return self.indices[empty]


def getTokenEmbeddingMatrices(corpus, attachedTokens=None):
    # load the pre-trained embeddings
    initType = configuration["model"]["embedding"]["initialisation"]["type"]
    relativeP = configuration["files"]["embedding"][initType]
    preTrainedEmb = dict()
    corpusTokens = set(getFreqDic(corpus).keys())
    if initType == "frWac200":
        preTrainedEmbModel = word2vec.load(
            os.path.join(configuration["path"]["projectPath"], relativeP))
        for item in preTrainedEmbModel.vocab:
            if item in corpusTokens:
                tokenIdxInVocab = np.where(preTrainedEmbModel.vocab == item)[0][0]
                preTrainedEmb[item] = preTrainedEmbModel.vectors[tokenIdxInVocab]
    elif initType == "frWiki50":
        with open(os.path.join(configuration["path"]["projectPath"], relativeP), 'r') as embF:
            for line in embF:
                if line.strip():
                    preTrainedEmb[line.split(' ')[0]] = line.split(' ')[1:-1]
    else:
        with open(os.path.join(configuration["path"]["projectPath"], relativeP), 'r') as embF:
            for line in embF:
                if line.strip():
                    preTrainedEmb[line.split(' ')[0]] = line.split(' ')[1:-1]
    if not configuration["xp"]["compo"]:
        indices = attachedTokens
    else:
        indices = getTokenVocabulary(corpus, preTrainedEmb)
    if initConf["active"] and initConf["token"]:
        embeddings = initialiseToken(indices, preTrainedEmb)
        return indices, embeddings
    return indices, None


#     # get a dictionary of train tokens with their frequencies (occurrences)
#     tokenFreqDic = getFreqDic(corpus)
#     # load the pre-trained embeddings
#     preTrainedEmb = word2vec.load(
#         os.path.join(configuration["path"]["projectPath"], configuration["files"]["embedding"]["frWac200"]))
#     # indices and embeddings are the reslut of the function
#     # indices is a dictionary mappng token => index; this index will help us in building
#     # weight matrix and in generating training data
#     indices, embeddings, idx, generatedRandomly, pretrained, testOnlyTokens, testOnlyTokensWithRandVect = \
#       dict(), dict(), 0, 0, 0, 0, 0
#     for sent in corpus.trainingSents + corpus.testingSents:
#         for token in sent.tokens:
#             tokenKey = token.getTokenOrLemma()
#             if tokenKey in indices:
#                 continue
#             if any(ch.isdigit() for ch in tokenKey):
#                 continue
#             # Token in the train data set
#             if tokenKey in tokenFreqDic:
#                 if not embConf["frequentTokens"] or tokenKey in corpus.mweTokenDictionary or \
#                         (embConf["frequentTokens"] and tokenFreqDic[tokenKey] > taux):
#                     if tokenKey in preTrainedEmb.vocab:
#                         tokenIdxInVocab = np.where(preTrainedEmb.vocab == tokenKey)[0][0]
#                         embeddings[tokenKey] = preTrainedEmb.vectors[tokenIdxInVocab]
#                         pretrained += 1
#                     else:
#                         embeddings[tokenKey] = getRandomVector(len(embeddings.values()[0]))
#                         generatedRandomly += 1
#                         # print tokenKey
#                     indices[tokenKey] = idx
#                     idx += 1
#             # Token belongs to test only
#             else:
#                 testOnlyTokens += 1
#                 if tokenKey in preTrainedEmb.vocab and uniform(0, 1) < configuration["constants"]["alpha"]:
#                     tokenIdxInVocab = np.where(preTrainedEmb.vocab == tokenKey)[0][0]
#                     embeddings[tokenKey] = preTrainedEmb.vectors[tokenIdxInVocab]
#                     pretrained += 1
#                     indices[tokenKey] = idx
#                     idx += 1
#     embeddings[unk] = getRandomVector(len(preTrainedEmb.vectors[0]))
#     indices[unk] = idx
#     idx += 1
#     embeddings[number] = getRandomVector(len(preTrainedEmb.vectors[0]))
#     indices[number] = idx
#     idx += 1
#     if not embConf["concatenation"]:
#         embeddings[empty] = getRandomVector(len(preTrainedEmb.vectors[0]))
#         indices[empty] = idx
#     sys.stdout.write('Filtered token frequency dic: {0}'.format(len(indices)))
#     return indices, embeddings


def getTokenVocabulary(corpus, preTrainedEmb):
    # get a dictionary of train tokens with their frequencies (occurrences)
    tokenFreqDic = getFreqDic(corpus)
    indices, idx, generatedRandomly = dict(), 0, 0
    for sent in corpus.trainingSents + corpus.testingSents:
        for token in sent.tokens:
            tokenKey = token.getTokenOrLemma()
            if tokenKey in indices:
                continue
            if any(ch.isdigit() for ch in tokenKey):
                continue
            # Token in the train data set
            if tokenKey in tokenFreqDic:
                if not embConf["frequentTokens"] or tokenKey in corpus.mweTokenDictionary or \
                        (embConf["frequentTokens"] and tokenFreqDic[tokenKey] > 1):
                    indices[tokenKey] = idx
                    idx += 1
            # Token belongs to test only
            else:
                if tokenKey in preTrainedEmb and uniform(0, 1) < configuration["constants"]["alpha"]:
                    indices[tokenKey] = idx
                    idx += 1
    indices[unk] = idx
    idx += 1
    indices[number] = idx
    idx += 1
    if not embConf["concatenation"]:
        indices[empty] = idx
    sys.stdout.write('Token vocabulary: {0}'.format(len(indices)))
    return indices


def initialiseToken(indices, preTrainedEmb):
    # weight matrix and in generating training data
    embeddings = dict()
    for tokenKey in indices:
        if tokenKey in preTrainedEmb:
            # tokenIdxInVocab = np.where(preTrainedEmb.vocab == tokenKey)[0][0]
            # embeddings[tokenKey] = preTrainedEmb.vectors[tokenIdxInVocab]
            embeddings[tokenKey] = preTrainedEmb[tokenKey]
        else:
            embeddings[tokenKey] = getRandomVector(len(preTrainedEmb.values()[0]))
    embeddings[unk] = getRandomVector(len(preTrainedEmb.values()[0]))
    embeddings[number] = getRandomVector(len(preTrainedEmb.values()[0]))
    if not embConf["concatenation"]:
        embeddings[empty] = getRandomVector(len(preTrainedEmb.values()[0]))
    return embeddings


def getPOSEmbeddingMatrices(corpus, attachedPos=None):
    if embConf["usePos"]:
        if not configuration["xp"]["compo"]:
            indices = attachedPos
        else:
            indices = getPOSVocabulary(corpus)
        if initConf["oneHotPos"]:
            embeddings = {}
            embConf["posEmb"] = len(indices)
            oneHotVectors = to_categorical(indices.values(), num_classes=len(indices))
            idx = 0
            for v in indices.values():
                kv = None
                for k in indices:
                    if indices[k] == v:
                        kv = k
                        break
                if kv:
                    embeddings[kv] = oneHotVectors[idx]
                    idx += 1
                else:
                    pass
            # embeddings = initialiseOneHotPOS(corpus, indices)
            return indices, embeddings
        elif initConf["active"] and initConf["pos"]:
            embeddings = initialisePOS(corpus, indices)
            return indices, embeddings
        return indices, None
    return None, None


def getPOSVocabulary(corpus):
    indices, idx = dict(), 0
    freqDic = getFreqDic(corpus, posTag=True)
    for elem in freqDic:
        if elem in freqDic and freqDic[elem] > 1:
            indices[elem.lower()] = idx
            idx += 1
    indices[unk] = idx
    if not embConf["concatenation"]:
        idx += 1
        indices[empty] = idx
    return indices
