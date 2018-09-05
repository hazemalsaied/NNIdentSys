from collections import Counter
from random import uniform

import keras
import numpy as np
from keras.layers import Input, Dense, Embedding, GRU, Dropout
from keras.models import Model
from keras.utils import to_categorical

from corpus import getTokens
from nonCompoVocabulary import attachTokens
from reports import seperator, doubleSep, tabs
from wordEmbLoader import unk, number, empty

enableCategorization = False


class Network:

    def __init__(self, corpus):
        global tokenVocab, posVocab
        tokenVocab, posVocab = getVocab(corpus)
        printVocabReport()
        inputLayers, concLayers = [], []
        inputToken = Input((3,))
        inputPos = Input((3,))
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
        t = s.initialTransition
        while t:
            if t.configuration.stack:
                tokens = getTokens(t.configuration.stack[-1])
                tokenTxt, posTxt = attachTokens(tokens)
                tokenCounter.update({tokenTxt: 1})
                posCounter.update({posTxt: 1})
            t = t.next
    if compact:
        for t in tokenCounter.keys():
            if t not in corpus.mweTokenDictionary or '_' not in t:
                del tokenCounter[t]
    else:
        for t in tokenCounter:
            if tokenCounter[t] == 1 and '_' not in t and uniform(0, 1) < 0.5:
                del tokenCounter[t]
    tokenCounter.update({unk: 1, number: 1, empty: 1})
    posCounter.update({unk: 1, empty: 1})

    return {w: i for i, w in enumerate(tokenCounter.keys())}, {w: i for i, w in enumerate(posCounter.keys())}


def getIdxs(conf):
    tokenIdxs, posIdxs = [], []
    if conf.stack:
        tokenTxt, posTxt = attachTokens(getTokens(conf.stack[-1]))
        tokenIdxs.append(tokenVocab[tokenTxt] if tokenTxt in tokenVocab else tokenVocab[unk])
        posIdxs.append(posVocab[posTxt] if posTxt in posVocab else posVocab[unk])
        if len(conf.stack) > 1:
            tokenTxt, posTxt = attachTokens(getTokens(conf.stack[-2]))
            tokenIdxs.append(tokenVocab[tokenTxt] if tokenTxt in tokenVocab else tokenVocab[unk])
            posIdxs.append(posVocab[posTxt] if posTxt in posVocab else posVocab[unk])
        else:
            tokenIdxs.append(tokenVocab[empty])
            posIdxs.append(posVocab[empty])
    else:
        tokenIdxs.append(tokenVocab[empty])
        tokenIdxs.append(tokenVocab[empty])
        posIdxs.append(posVocab[empty])
        posIdxs.append(posVocab[empty])

    if conf.buffer:
        t = conf.buffer[0]
        tokenTxt, posTxt = t.getTokenOrLemma(), t.posTag.lower()
        tokenIdxs.append(tokenVocab[tokenTxt] if tokenTxt in tokenVocab else tokenVocab[unk])
        posIdxs.append(posVocab[posTxt] if posTxt in posVocab else posVocab[unk])
    else:
        tokenIdxs.append(tokenVocab[empty])
        posIdxs.append(posVocab[empty])
    return np.asarray(tokenIdxs), np.asarray(posIdxs)


def printVocabReport():
    res = seperator + tabs + 'Vocabulary' + doubleSep
    res += tabs + 'Tokens := {0} * POS : {1}'.format(len(tokenVocab), len(posVocab))
    res += seperator
    return res
