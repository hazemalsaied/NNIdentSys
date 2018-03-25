import logging
import os
from random import uniform

import gensim
import numpy as np
import word2vec

from config import configuration
from corpus import getTokens

unk = configuration["constants"]["unk"]
empty = configuration["constants"]["empty"]
number = configuration["constants"]["number"]


class Vocabulary:
    def __init__(self, corpus, taux=1):

        self.attachedTokens, self.attachedPos = self.getAttachedVoc(corpus)
        print 'self.attachedTokens:', len(self.attachedTokens)
        print 'self.attachedPos:', len(self.attachedPos)
        global embConf
        embConf = configuration["model"]["embedding"]
        self.posIndices, self.posEmbeddings = getPOSEmbeddingMatrices(corpus, embConf["posEmb"], taux=taux)
        self.postagDim = len(self.posEmbeddings.values()[0])

        self.tokenIndices, self.tokenEmbeddings = getTokenEmbeddingMatrices(corpus, taux=taux)
        self.tokenDim = len(self.tokenEmbeddings.values()[0]) if embConf["initialisation"] else embConf["tokenEmb"]

        self.indices, self.embeddings = self.getEmbeddingMatrices(corpus)
        self.embDim = self.tokenDim + self.postagDim if embConf["concatenation"] and embConf["pos"] else self.tokenDim
        self.size = len(self.indices)
        logging.warn('Vocabulary size: {0}'.format(self.size))

    def getEmbeddingMatrices(self, corpus):
        indices, embeddings, idx = self.generateUnknownKeys()
        for sent in corpus.trainingSents + corpus.testingSents:
            for token in sent.tokens:
                tokenKey = token.getTokenOrLemma() if token.getTokenOrLemma() in self.tokenIndices else unk
                key = tokenKey
                if embConf["pos"]:
                    posKey = token.posTag.lower() if token.posTag.lower() in self.posIndices else unk
                    key += '_' + posKey
                if key not in indices:
                    if embConf["pos"]:
                        embeddings[key] = np.concatenate((self.tokenEmbeddings[tokenKey],
                                                          self.posEmbeddings[posKey]))
                    else:
                        embeddings[key] = self.tokenEmbeddings[tokenKey]

                    indices[key] = idx
                    idx += 1
        return indices, embeddings

    def generateUnknownKeys(self):
        indices, embeddings, idx = dict(), dict(), 0
        if embConf["pos"]:
            for posTag in self.posIndices.keys():
                key = unk + '_' + posTag.lower()
                if key not in indices:
                    embeddings[key] = np.concatenate(
                        (self.tokenEmbeddings[unk], self.posEmbeddings[posTag.lower()]))
                    indices[key] = idx
                    idx += 1
                key1 = number + '_' + posTag.lower()
                if key1 not in indices:
                    embeddings[key1] = np.concatenate(
                        (self.tokenEmbeddings[number], self.posEmbeddings[posTag.lower()]))
                    indices[key1] = idx
                    idx += 1
        key = unk
        if embConf["pos"]:
            key += '_' + unk
        if key not in indices:
            if embConf["pos"]:
                embeddings[key] = np.concatenate((self.tokenEmbeddings[unk], self.posEmbeddings[unk]))
            else:
                embeddings[key] = self.tokenEmbeddings[unk]
            indices[key] = idx
            idx += 1
        key1 = number
        if embConf["pos"]:
            key1 += '_' + unk
        if key1 not in indices:
            if embConf["pos"]:
                embeddings[key1] = np.concatenate((self.tokenEmbeddings[number], self.posEmbeddings[unk]))
            else:
                embeddings[key1] = self.tokenEmbeddings[unk]
            indices[key1] = idx
            idx += 1
        if embConf["pos"]:
            embeddings[empty] = np.zeros((len(self.tokenEmbeddings.values()[0]) + len(self.posEmbeddings.values()[0])))
        else:
            embeddings[empty] = np.zeros((len(self.tokenEmbeddings.values()[0])))
        indices[empty] = idx
        idx += 1
        return indices, embeddings, idx

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
                tokenTxt = token.text.lower()
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
                    tokens = getTokens(trans.configuration.stack[0])
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
        for k in tokenVocab.keys():
            if tokenVocab[k] == 1 and '_' not in k:
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
            tokenTxt += t.text.lower() + '_'
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


def getTokenEmbeddingMatrices(corpus, taux):
    # get a dictionary of train tokens with their frequencies (occurrences)
    tokenFreqDic = getFreqDic(corpus)
    # load the pre-trained embeddings
    preTrainedEmb = word2vec.load(
        os.path.join(configuration["path"]["projectPath"], configuration["files"]["embedding"]["frWac200"]))
    # indices and embeddings are the reslut of the function
    # indices is a dictionary mappng token => index; this index will help us in building
    # weight matrix and in generating training data
    indices, embeddings, idx, generatedRandomly, pretrained, testOnlyTokens, testOnlyTokensWithRandVect = dict(), dict(), 0, 0, 0, 0, 0
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
                        (embConf["frequentTokens"] and tokenFreqDic[tokenKey] > taux):
                    if tokenKey in preTrainedEmb.vocab:
                        tokenIdxInVocab = np.where(preTrainedEmb.vocab == tokenKey)[0][0]
                        embeddings[tokenKey] = preTrainedEmb.vectors[tokenIdxInVocab]
                        pretrained += 1
                    else:
                        embeddings[tokenKey] = getRandomVector(len(embeddings.values()[0]))
                        generatedRandomly += 1
                        # print tokenKey
                    indices[tokenKey] = idx
                    idx += 1
            # Token belongs to test only
            else:
                testOnlyTokens += 1
                if tokenKey in preTrainedEmb.vocab and uniform(0, 1) < configuration["constants"]["alpha"]:
                    tokenIdxInVocab = np.where(preTrainedEmb.vocab == tokenKey)[0][0]
                    embeddings[tokenKey] = preTrainedEmb.vectors[tokenIdxInVocab]
                    pretrained += 1
                    indices[tokenKey] = idx
                    idx += 1
    embeddings[unk] = getRandomVector(len(preTrainedEmb.vectors[0]))
    indices[unk] = idx
    idx += 1
    embeddings[number] = getRandomVector(len(preTrainedEmb.vectors[0]))
    indices[number] = idx
    idx += 1
    if not embConf["concatenation"]:
        embeddings[empty] = getRandomVector(len(preTrainedEmb.vectors[0]))
        indices[empty] = idx
    logging.warn('Filtered token frequency dic: {0}'.format(len(indices)))
    return indices, embeddings


def getPOSEmbeddingMatrices(corpus, dimension, taux, window=3):
    indices, embeddings, idx = dict(), dict(), 0
    freqDic = getFreqDic(corpus, posTag=True)
    traindEmb = trainPosEmbWithWordToVec(corpus, dimension, window)
    for elem in traindEmb.wv.vocab:
        if elem.lower() in freqDic and freqDic[elem] > taux:
            embeddings[elem.lower()] = traindEmb.wv[elem]
            indices[elem.lower()] = idx
            idx += 1
    embeddings[unk] = getRandomVector(dimension)
    indices[unk] = idx
    idx += 1
    if not embConf["concatenation"]:
        embeddings[empty] = getRandomVector(dimension)
        indices[empty] = idx
    logging.warn('{0} Pos tags'.format(len(traindEmb.wv.vocab) + 1))
    return indices, embeddings


def getFreqDic(corpus, posTag=False):
    freqDic = dict()
    for sent in corpus.trainingSents:
        for token in sent.tokens:
            key = token.posTag.lower() if posTag else token.getTokenOrLemma().lower()
            if key not in freqDic:
                freqDic[key] = 1
            else:
                freqDic[key] += 1
    logging.warn('{0} frequency dic: {1}'.format('posTag' if posTag else 'token', len(freqDic)))
    return freqDic


def trainPosEmbWithWordToVec(corpus, dimension, window=3):
    normailsedSents = []
    for sent in corpus.trainingSents + corpus.testingSents:
        normailsedSent = []
        for token in sent.tokens:
            normailsedSent.append(token.posTag.lower())
        normailsedSents.append(normailsedSent)
    model = gensim.models.Word2Vec(normailsedSents, size=dimension, window=window)
    return model


def getRandomVector(length):
    return np.asarray([float(val) for val in np.random.uniform(low=-0.01, high=0.01, size=int(length))])
