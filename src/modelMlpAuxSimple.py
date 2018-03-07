import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from numpy import argmax
from keras.models import Sequential
import settings
from model import Normalizer, Network
from transitions import TransitionType

PADDING_ON_S0 = 3
PADDING_ON_S1 = 2
PADDING_ON_B0 = 2
INPUT_WORDS = PADDING_ON_S0 + PADDING_ON_S1 + PADDING_ON_B0


class NetworkMLPAuxSimple(Network):
    def __init__(self, normalizer):
        unitNum = normalizer.nnExtractor.featureNum
        self.model = Sequential()
        self.model.add(Dense(1024, activation='relu',  input_dim=unitNum))
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
