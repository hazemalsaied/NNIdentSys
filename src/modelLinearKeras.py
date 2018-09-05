import datetime

import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import to_categorical
from numpy import argmax

import compoVocabulary
import nonCompoVocabulary
import reports
from corpus import getTokens
from reports import *
from transitions import TransitionType


class LinearKerasModel:
    def __init__(self, vocabSize):
        inputs = Input(shape=(vocabSize * tokenNum,))
        x = Dense(8, activation='softmax')(inputs)
        # predictions = Dense(10, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=x)

        # self.model = Sequential()
        # self.model.add(Dense(100, input_dim=vocabSize * tokenNum))
        # self.model.add(Dense(len(TransitionType), activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def predict(self, trans, normalizer):
        dataEntry = normalizer.vectorize(trans.configuration)
        dataEntry = np.asarray(dataEntry)
        dataEntry = np.reshape(dataEntry, (1, len(dataEntry)))
        oneHotRep = self.model.predict(dataEntry, batch_size=1,
                                       verbose=configuration["model"]["predict"]["verbose"])
        return argmax(oneHotRep)


def train(model, corpus, normaliser):
    trainConf = configuration["model"]["train"]
    labels, data = generateLearningData(corpus, normaliser)
    data = np.asarray(data)
    labels = to_categorical(labels, num_classes=len(TransitionType))
    model.fit(data, labels.toarray(), epochs=trainConf["epochs"],
              batch_size=trainConf["batchSize"],
              verbose=trainConf["verbose"])


s0Padding = 4
s1Padding = 4
bPadding = 2
tokenNum = s0Padding + s1Padding + bPadding
unk = configuration["constants"]["unk"]
empty = configuration["constants"]["empty"]
number = configuration["constants"]["number"]


class Normalizer:
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        if configuration["xp"]["compo"]:
            self.vocabulary = compoVocabulary.Vocabulary(corpus)
        else:
            self.vocabulary = nonCompoVocabulary.Vocabulary(corpus)
        self.tokens = self.vocabulary.tokenIndices.keys()
        self.pos = self.vocabulary.posIndices.keys()
        self.tokens.append(unk)
        del self.vocabulary

    def vectorize(self, config):
        s0Tokens, s1Tokens, bTokens = [], [], []
        if config.stack:
            s0Tokens = getTokens(config.stack[-1])
            if len(config.stack) > 1:
                s1Tokens = getTokens(config.stack[-2])
        if config.buffer:
            bTokens = getTokens(config.buffer[:2])
        s0Tokens = s0Tokens[-1 * s0Padding:]
        while len(s0Tokens) < s0Padding:
            s0Tokens = [None] + s0Tokens
        s1Tokens = s1Tokens[-1 * s1Padding:]
        while len(s1Tokens) < s1Padding:
            s1Tokens = [None] + s1Tokens
        while len(bTokens) < bPadding:
            bTokens.append(None)
        tokens = s0Tokens + s1Tokens + bTokens
        res = []
        for t in tokens:
            res.append(self.getTokenVector(t))
            res.append(self.getPosVector(t))
        res = np.concatenate((res))
        return res

    def getPosVector(self, token):
        if token:
            posKey = token.posTag
            if posKey not in self.pos:
                posKey = unk
            if posKey in self.pos:
                posIdx = self.pos.index(posKey)
                res = np.zeros(len(self.pos))
                res[posIdx] = 1
                return res
        return np.zeros(len(self.pos))

    def getTokenVector(self, token):
        if token:
            tokenKey = token.text.lower()
            if tokenKey not in self.tokens:
                if any(ch.isdigit() for ch in token.getTokenOrLemma()):
                    tokenKey = number
                else:
                    tokenKey = unk
            if tokenKey in self.tokens:
                tokenIdx = self.tokens.index(tokenKey)
                res = np.zeros(len(self.tokens))
                res[tokenIdx] = 1
                return res
        return np.zeros(len(self.tokens))


def generateLearningData(corpus, normaliser):
    data, labels = [], []
    for sent in corpus:
        trans = sent.initialTransition
        while trans.next:
            dataEntry = normaliser.vectorize(trans.configuration)
            data.append(dataEntry)
            labels = np.append(labels, trans.next.type.value)
            trans = trans.next
    return np.asarray(labels), data
