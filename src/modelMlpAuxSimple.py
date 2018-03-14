import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from numpy import argmax

from config import configuration
from model import AbstractNormalizer, AbstractNetwork
from transitions import TransitionType


INPUT_WORDS = configuration["model"]["embedding"]["s0Padding"] + configuration["model"]["embedding"]["s1Padding"] + configuration["model"]["embedding"]["bPadding"]


class Networ(AbstractNetwork):
    def __init__(self, normalizer):
        unitNum = normalizer.nnExtractor.featureNum
        self.model = Sequential()
        self.model.add(Dense(1024, activation='relu', input_dim=unitNum))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(len(TransitionType), activation='softmax'))
        super(Networ, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        dataEntry = np.reshape(dataEntry, (1, len(dataEntry[0])))
        oneHotRep = self.model.predict(dataEntry, batch_size=1, verbose=configuration["model"]["predict"]["verbose"])
        # oneHotRep = self.model.predict(np.asarray([dataEntry]), batch_size=1, verbose=PREDICT_VERBOSE)
        return argmax(oneHotRep)


class Normalizer(AbstractNormalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(Normalizer, self).__init__(corpus, 1, True)

    def normalize(self, trans):
        dataEntry4 = self.nnExtractor.vectorize(trans)
        return [np.asarray(dataEntry4)]
