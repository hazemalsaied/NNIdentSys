import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from numpy import argmax

import settings
from model import Normalizer, Network
from transitions import TransitionType


INPUT_WORDS = settings.PADDING_ON_S0 + settings.PADDING_ON_S1 + settings.PADDING_ON_B0


class NetworkMLPAuxSimple(Network):
    def __init__(self, normalizer):
        unitNum = normalizer.nnExtractor.featureNum
        self.model = Sequential()
        self.model.add(Dense(1024, activation='relu', input_dim=unitNum))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(len(TransitionType), activation='softmax'))
        super(NetworkMLPAuxSimple, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        dataEntry = np.reshape(dataEntry, (1, len(dataEntry[0])))
        oneHotRep = self.model.predict(dataEntry, batch_size=1, verbose=settings.NN_PREDICT_VERBOSE)
        # oneHotRep = self.model.predict(np.asarray([dataEntry]), batch_size=1, verbose=PREDICT_VERBOSE)
        return argmax(oneHotRep)


class NormalizerMLPAuxSimple(Normalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(NormalizerMLPAuxSimple, self).__init__(corpus, 1, True)

    def normalize(self, trans):
        dataEntry4 = self.nnExtractor.vectorize(trans)
        return [np.asarray(dataEntry4)]
