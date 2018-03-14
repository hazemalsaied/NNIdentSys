import keras
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

from config import configuration
from corpus import getTokens
from model import AbstractNetwork, AbstractNormalizer
from vocabulary import empty

embConf = configuration["model"]["embedding"]
INPUT_WORDS = embConf["s0Padding"] + embConf["s1Padding"] + \
              embConf["bPadding"]


class Network(AbstractNetwork):
    def __init__(self, normalizer):

        if not embConf["concatenation"]:
            posLayer, posFlattenLayer = AbstractNetwork.createPOSEmbeddingModule(INPUT_WORDS, normalizer)
            tokenLayer, tokenFlattenLayer = AbstractNetwork.createTokenEmbeddingModule(INPUT_WORDS, normalizer)
            flattenLayer = keras.layers.concatenate([posFlattenLayer, tokenFlattenLayer])
        else:
            wordLayer, flattenLayer = AbstractNetwork.createEmbeddingModule(INPUT_WORDS, normalizer)
        # Auxiliary feature vectors
        auxFeatureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='auxFeatureLayer')
        # Merge layer
        concLayer = keras.layers.concatenate([flattenLayer, auxFeatureLayer])
        # MLP module
        mainOutputLayer = self.createMLPModule(concLayer)
        if not embConf["concatenation"]:
            self.model = Model(inputs=[posLayer, tokenLayer, auxFeatureLayer], outputs=mainOutputLayer)
        else:
            self.model = Model(inputs=[wordLayer, auxFeatureLayer], outputs=mainOutputLayer)
        super(Network, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans,
                                         seperatedModules=not embConf["concatenation"])
        if not embConf["concatenation"]:
            inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]]), np.asarray([dataEntry[2]])]
        else:
            inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]])]
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=configuration["model"]["predict"]["verbose"])
        return argmax(oneHotRep)


class Normalizer(AbstractNormalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def normalize(self, trans, useEmbedding=False, useConcatenation=False, usePos=False, useFeatures=False):
        if useFeatures:
            features = np.asarray(self.nnExtractor.vectorize(trans))
            if useEmbedding:
                if useConcatenation:
                    words = self.getWordIndices(trans)
                    return [words, features]
                else:
                    tokens = self.getTokenINdices(trans)
                    if usePos:
                        pos = self.getPosIndices(trans)
                        return [pos, tokens, features]
                    else:
                        return [tokens, features]
            else:
                return [features]
        elif useEmbedding:
            if useConcatenation:
                words = self.getWordIndices(trans)
                return [words]
            else:
                tokens = self.getTokenINdices(trans)
                if usePos:
                    pos = self.getPosIndices(trans)
                    return [pos, tokens]
                else:
                    return [tokens]
        return []

    def getWordIndices(self, trans):
        s0elems, s1elems, belems = [], [], []
        emptyIdx = self.vocabulary.indices[empty]
        if trans.configuration.stack:
            s0elems = self.getIndices(getTokens(trans.configuration.stack[-1]))
            if len(trans.configuration.stack) > 1:
                s1elems = self.getIndices(getTokens(trans.configuration.stack[-2]))
        s0elems = padSequence(s0elems, "s0Padding", emptyIdx)
        s1elems = padSequence(s1elems, "s1Padding", emptyIdx)
        if trans.configuration.buffer:
            belems = self.getIndices(trans.configuration.buffer[:2])
        belems = padSequence(belems, "bPadding", emptyIdx)
        words = np.concatenate((s0elems, s1elems, belems), axis=0)
        return words

    def getPosIndices(self, trans):
        s0Pos, s1Pos, bPos, = [], [], []
        posEmptyIdx = self.vocabulary.posIndices[empty]
        if trans.configuration.stack:
            s0Pos = self.getIndices(getTokens(trans.configuration.stack[-1]), getPos=True)
            if len(trans.configuration.stack) > 1:
                s1Pos = self.getIndices(getTokens(trans.configuration.stack[-2]), getPos=True)
        s0Pos = padSequence(s0Pos, "s0Padding", posEmptyIdx)
        s1Pos = padSequence(s1Pos, "s1Padding", posEmptyIdx)
        if trans.configuration.buffer:
            bPos = self.getIndices(trans.configuration.buffer[:2], getPos=True)
        bPos = padSequence(bPos, "bPadding", posEmptyIdx)
        pos = np.concatenate((s0Pos, s1Pos, bPos), axis=0)
        return pos

    def getTokenINdices(self, trans):
        s0Tokens, s1Tokens, bTokens = [], [], []
        tokenEmptyIdx = self.vocabulary.tokenIndices[empty]
        if trans.configuration.stack:
            s0Tokens = self.getIndices(getTokens(trans.configuration.stack[-1]), getToken=True)
            if len(trans.configuration.stack) > 1:
                s1Tokens = self.getIndices(getTokens(trans.configuration.stack[-2]), getToken=True)
        s0Tokens = padSequence(s0Tokens, "s0Padding", tokenEmptyIdx)
        s1Tokens = padSequence(s1Tokens, "s1Padding", tokenEmptyIdx)
        if trans.configuration.buffer:
            bTokens = self.getIndices(trans.configuration.buffer[:2], getToken=True)
        bTokens = padSequence(bTokens, "bPadding", tokenEmptyIdx)
        tokens = np.concatenate((s0Tokens, s1Tokens, bTokens), axis=0)
        return tokens


def padSequence(seq, label, emptyIdx):
    embConf = configuration["model"]["embedding"]
    return np.asarray(pad_sequences([seq], maxlen=embConf[label], value=emptyIdx))[0]
