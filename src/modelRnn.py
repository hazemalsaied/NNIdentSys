from collections import Counter
from random import uniform

import keras
import numpy as np
from keras.layers import Input, Dense, Embedding, GRU, Dropout
from keras.models import Model
from keras.utils import to_categorical

from corpus import getTokens
from wordEmbLoader import unk, number, empty

enableCategorization = False


class Network:

    def __init__(self, corpus):
        global tokenVocab, posVocab
        tokenVocab, posVocab = getVocab(corpus)

        inputLayers, concLayers = [], []
        inputToken = Input((7,))
        inputPos = Input((7,))
        inputLayers.append(inputToken)
        inputLayers.append(inputPos)
        tokenEmb, posEmb = 48, 12
        tokenEmb = Embedding(len(tokenVocab), tokenEmb)(inputToken)
        tokenRnn = GRU(16)(tokenEmb)
        concLayers.append(tokenRnn)
        posEmb = Embedding(len(posVocab), posEmb)(inputPos)
        posRnn = GRU(4)(posEmb)
        concLayers.append(posRnn)
        conc = keras.layers.concatenate(concLayers)
        dense1Layer = Dense(32, activation='relu')(conc)
        lastLayer = Dropout(0.3)(dense1Layer)
        # dense2Layer = Dense(16, activation='relu')(lastLayer)
        # lastLayer = Dropout(0.2)(dense2Layer)
        softmax = Dense(8 if enableCategorization else 4, activation='softmax')(lastLayer)
        self.model = Model(inputs=inputLayers, outputs=softmax)
        print self.model.summary()

    def predict(self, trans):
        tokenIdxs, posIdxs = getIdxs(trans.configuration)
        oneHotRep = self.model.predict([np.asarray([tokenIdxs]), np.asarray([posIdxs])], batch_size=1)
        return oneHotRep[0]


def getLearningData(corpus):
    lbls, tokenIdxs, posIdxs = [], [], []
    for s in corpus.trainingSents:
        t = s.initialTransition
        while t and t.next:
            idxs = getIdxs(t.configuration)
            tokenIdxs.append(idxs[0])
            posIdxs.append(idxs[1])
            np.append(lbls, t.next.type.value)
            lbl = 3 if t.next.type.value > 2 and not enableCategorization else t.next.type.value
            lbls.append(lbl)
            t = t.next
    lbls = to_categorical(lbls, num_classes=8 if enableCategorization else 4)
    return lbls, [np.asarray(tokenIdxs), np.asarray(posIdxs)]


def train(cls, corpus):
    labels, data = getLearningData(corpus)
    optimizer = keras.optimizers.Adagrad(lr=0.02, epsilon=None, decay=0.0)
    cls.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    cls.model.fit(data, labels,
                  validation_split=.1,
                  epochs=100,
                  batch_size=64,
                  verbose=2)


def getVocab(corpus, compact=True):
    tokenCounter, posCounter = Counter(), Counter()
    for s in corpus.trainingSents:
        for t in s.tokens:
            tokenCounter.update({t.getTokenOrLemma(): 1})
            posCounter.update({t.posTag.lower(): 1})
    if compact:
        for t in tokenCounter.keys():
            if t not in corpus.mweTokenDictionary:
                del tokenCounter[t]
    else:
        for t in tokenCounter:
            if tokenCounter[t] == 1 and uniform(0, 1) < 0.5:
                del tokenCounter[t]
    tokenCounter.update({unk: 1, number: 1, empty: 1})
    posCounter.update({unk: 1, empty: 1})

    return {w: i for i, w in enumerate(tokenCounter.keys())}, {w: i for i, w in enumerate(posCounter.keys())}


def getIdxs(config):
    idxs, posIdxs = [], []
    global tokenVocab, posVocab
    if config.stack and len(config.stack) > 1:
        for t in getTokens(config.stack[-2])[:2]:
            if t.isNumber():
                idxs.append(tokenVocab[number])
            else:
                idxs.append(tokenVocab[t.getTokenOrLemma()] if t.getTokenOrLemma() in tokenVocab else tokenVocab[unk])
            posIdxs.append(posVocab[t.posTag.lower()] if t.posTag.lower() in posVocab else posVocab[unk])
    while len(idxs) < 2:
        idxs = idxs + [tokenVocab[empty]]
        posIdxs = posIdxs + [posVocab[empty]]

    if config.stack:
        for t in getTokens(config.stack[-1])[:4]:
            if t.isNumber():
                idxs.append(tokenVocab[number])
            else:
                idxs.append(tokenVocab[t.getTokenOrLemma()] if t.getTokenOrLemma() in tokenVocab else tokenVocab[unk])
            posIdxs.append(posVocab[t.posTag.lower()] if t.posTag.lower() in posVocab else posVocab[unk])
    while len(idxs) < 6:
        idxs = idxs + [tokenVocab[empty]]
        posIdxs = posIdxs + [posVocab[empty]]

    if config.buffer:
        t = config.buffer[0]
        if t.isNumber():
            idxs.append(tokenVocab[number])
        else:
            idxs.append(tokenVocab[t.getTokenOrLemma()] if t.getTokenOrLemma() in tokenVocab else tokenVocab[unk])
        posIdxs.append(posVocab[t.posTag.lower()] if t.posTag.lower() in posVocab else posVocab[unk])
    else:
        idxs = idxs + [tokenVocab[empty]]
        posIdxs = posIdxs + [posVocab[empty]]
    return np.asarray(idxs), np.asarray(posIdxs)
