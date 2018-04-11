import numpy as np
from keras.preprocessing.sequence import pad_sequences

import reports
from corpus import getTokens
from extraction import Extractor
from reports import *
from vocabulary import Vocabulary, empty


class Normalizer:
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        global embConf, initConf
        embConf = configuration["model"]["embedding"]
        initConf = embConf["initialisation"]
        self.nnExtractor, self.vocabulary, self.posWeightMatrix, self.weightMatrix, self.tokenWeightMatrix = \
            None, None, None, None, None
        if embConf["active"]:
            self.vocabulary = Vocabulary(corpus)
            if initConf["active"]:
                if embConf["concatenation"]:
                    self.weightMatrix = initMatrix(self.vocabulary)
                    del self.vocabulary.embeddings
                else:
                    if embConf["usePos"]:
                        self.posWeightMatrix = initMatrix(self.vocabulary, usePos=True)
                        del self.vocabulary.posEmbeddings

                    self.tokenWeightMatrix = initMatrix(self.vocabulary, useToken=True)
                    del self.vocabulary.tokenEmbeddings
        if configuration["features"]["active"]:
            self.nnExtractor = Extractor(corpus)
        reports.saveNormalizer(self)

    def generateLearningData(self, corpus):
        embConf = configuration["model"]["embedding"]
        data, labels = [], []
        useEmbedding = embConf["active"]
        useConcatenation = embConf["concatenation"]
        usePos = embConf["usePos"]
        useFeatures = configuration["features"]["active"]
        data = None
        for sent in corpus:
            trans = sent.initialTransition
            while trans.next:
                dataEntry = self.normalize(trans, useConcatenation=useConcatenation, useEmbedding=useEmbedding,
                                           usePos=usePos, useFeatures=useFeatures)
                # First iteration only
                if not data:
                    data = []
                    self.inputListDimension = len(dataEntry)
                    if len(dataEntry) > 1:
                        for i in range(len(dataEntry)):
                            data.append(list())
                for i in range(self.inputListDimension):
                    if self.inputListDimension != 1:
                        data[i].append(dataEntry[i])
                    else:
                        data.append(dataEntry[i])
                labels = np.append(labels, trans.next.type.value)
                trans = trans.next
        if self.inputListDimension != 1:
            for i in range(len(data)):
                data[i] = np.asarray(data[i])
        if self.inputListDimension == 1:
            data = np.asarray(data)
        return np.asarray(labels), data

    def generateLearningDataAttached(self, corpus):
        labels, tokenData, posData, featureData = [], [], [], []
        useFeatures = configuration["features"]["active"]
        for sent in corpus.trainingSents:
            trans = sent.initialTransition
            while trans.next:
                tokenIdxs, posIdxs = self.getAttachedIndices(trans)
                tokenData.append(np.asarray(tokenIdxs))
                posData.append(np.asarray(posIdxs))
                # data.append(np.asarray([tokenIdxs, posIdxs]))
                if useFeatures:
                    features = np.asarray(self.nnExtractor.vectorize(trans))
                    featureData.append(np.asarray(features))
                labels = np.append(labels, trans.next.type.value)
                trans = trans.next
        if useFeatures:
            if configuration["model"]["embedding"]["usePos"]:
                return np.asarray(labels), [np.asarray(tokenData), np.asarray(posData), np.asarray(featureData)]
            else:
                return np.asarray(labels), [np.asarray(tokenData),  np.asarray(featureData)]
        if configuration["model"]["embedding"]["usePos"]:
            return np.asarray(labels), [np.asarray(tokenData), np.asarray(posData)]
        else:
            return np.asarray(labels), np.asarray(tokenData)

    def normalize(self, trans, useEmbedding=False, useConcatenation=False, usePos=False, useFeatures=False):
        results = []
        if useEmbedding:
            if useConcatenation:
                words = self.getIndices(trans)
                results.append(words)
            else:
                if usePos:
                    pos = self.getIndices(trans, getPos=True)
                    results.append(pos)
                tokens = self.getIndices(trans, getToken=True)
                results.append(tokens)
        if useFeatures:
            features = np.asarray(self.nnExtractor.vectorize(trans))
            results.append(features)
        return results

    def getIndices(self, trans, getPos=False, getToken=False):
        s0elems, s1elems, belems = [], [], []
        emptyIdx = self.vocabulary.getEmptyIdx(getPos=getPos, getToken=getToken)
        if trans.configuration.stack:
            s0Tokens = getTokens(trans.configuration.stack[-1])
            s0elems = self.vocabulary.getIndices(s0Tokens, getPos=getPos, getToken=getToken)
            if len(trans.configuration.stack) > 1:
                s1Tokens = getTokens(trans.configuration.stack[-2])
                s1elems = self.vocabulary.getIndices(s1Tokens, getPos=getPos, getToken=getToken)
        s0elems = padSequence(s0elems, "s0Padding", emptyIdx)
        s1elems = padSequence(s1elems, "s1Padding", emptyIdx)
        if trans.configuration.buffer:
            bTokens = trans.configuration.buffer[:2]
            belems = self.vocabulary.getIndices(bTokens, getPos=getPos, getToken=getToken)
        belems = padSequence(belems, "bPadding", emptyIdx)
        words = np.concatenate((s0elems, s1elems, belems), axis=0)
        return words

    def getAttachedIndices(self, trans):
        emptyTokenIdx = self.vocabulary.attachedTokens[empty]
        emptyPosIdx = self.vocabulary.attachedPos[empty]
        tokenIdxs, posIdxs = [], []
        if trans.configuration.stack:
            s0Tokens = getTokens(trans.configuration.stack[-1])
            tokenIdx, posIdx = self.vocabulary.getAttachedIndices(s0Tokens)
            tokenIdxs.append(tokenIdx)
            posIdxs.append(posIdx)
            if len(trans.configuration.stack) > 1:
                s1Tokens = getTokens(trans.configuration.stack[-2])
                tokenIdx, posIdx = self.vocabulary.getAttachedIndices(s1Tokens)
                tokenIdxs.append(tokenIdx)
                posIdxs.append(posIdx)
            else:
                tokenIdxs.append(emptyTokenIdx)
                posIdxs.append(emptyPosIdx)
        else:
            tokenIdxs.append(emptyTokenIdx)
            tokenIdxs.append(emptyTokenIdx)
            posIdxs.append(emptyPosIdx)
            posIdxs.append(emptyPosIdx)

        if trans.configuration.buffer:
            tokenIdx, posIdx = self.vocabulary.getAttachedIndices([trans.configuration.buffer[0]])
            tokenIdxs.append(tokenIdx)
            posIdxs.append(posIdx)
            if len(trans.configuration.buffer) > 1:
                tokenIdx, posIdx = self.vocabulary.getAttachedIndices([trans.configuration.buffer[1]])
                tokenIdxs.append(tokenIdx)
                posIdxs.append(posIdx)
            else:
                tokenIdxs.append(emptyTokenIdx)
                posIdxs.append(emptyPosIdx)
        else:
            tokenIdxs.append(emptyTokenIdx)
            tokenIdxs.append(emptyTokenIdx)
            posIdxs.append(emptyPosIdx)
            posIdxs.append(emptyPosIdx)

        return np.asarray(tokenIdxs), np.asarray(posIdxs)


def padSequence(seq, label, emptyIdx):
    embConf = configuration["model"]["padding"]
    return np.asarray(pad_sequences([seq], maxlen=embConf[label], value=emptyIdx))[0]


def initMatrix(vocabulary, usePos=False, useToken=False):
    if usePos:
        if not configuration["model"]["embedding"]["initialisation"]["pos"]:
            return None
        indices = vocabulary.posIndices
        dim = embConf["posEmb"]
        embeddings = vocabulary.posEmbeddings
    elif useToken:
        if not configuration["model"]["embedding"]["initialisation"]["token"]:
            return None
        indices = vocabulary.tokenIndices
        dim = embConf["tokenEmb"]
        embeddings = vocabulary.tokenEmbeddings
    else:
        indices = vocabulary.indices
        dim = vocabulary.embDim
        embeddings = vocabulary.embeddings

    matrix = np.zeros((len(indices), dim))
    for elem in indices:
        matrix[indices[elem]] = embeddings[elem]
    return matrix
