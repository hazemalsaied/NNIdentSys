import numpy as np
from keras.layers import Input, Embedding, Flatten
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

import settings
from corpus import getTokens
from model import Normalizer, Network
from vocabulary import empty

PADDING_ON_S0 = 3
PADDING_ON_S1 = 2
PADDING_ON_B0 = 2
INPUT_WORDS = PADDING_ON_S0 + PADDING_ON_S1 + PADDING_ON_B0


class NetworkMLPHyperSimple(Network):
    def __init__(self, normalizer):
        # Buffer-based Embedding Module
        wordLayer = Input(shape=(INPUT_WORDS,), name='words')
        # self.model = Sequential()
        sharedEmbedding = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                    weights=[normalizer.weightMatrix], trainable=True)(wordLayer)
        bOutputLayer = Flatten(name='buffOutputLayer')(sharedEmbedding)
        # MLP module
        mainOutputLayer = self.createMLPModule(bOutputLayer)
        self.model = Model(inputs=wordLayer, outputs=mainOutputLayer)

        super(NetworkMLPHyperSimple, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        dataEntry = np.reshape(dataEntry, (1, len(dataEntry)))
        oneHotRep = self.model.predict(dataEntry, batch_size=1, verbose=settings.NN_PREDICT_VERBOSE)
        # oneHotRep = self.model.predict(np.asarray([dataEntry]), batch_size=1, verbose=PREDICT_VERBOSE)
        return argmax(oneHotRep)


class NormalizerMLPHyperSimple(Normalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(NormalizerMLPHyperSimple, self).__init__(corpus, 1, False)

    def normalize(self, trans):
        dataEntry1, dataEntry2, dataEntry3 = [], [], []
        if trans.configuration.stack:
            dataEntry1 = self.getIndices(getTokens(trans.configuration.stack[-1]))
            if len(trans.configuration.stack) > 1:
                dataEntry2 = self.getIndices(getTokens(trans.configuration.stack[-2]))
        if trans.configuration.buffer:
            dataEntry3 = self.getIndices(trans.configuration.buffer[:2])
        emptyIdx = self.vocabulary.indices[empty]

        dataEntry1 = np.asarray(pad_sequences([dataEntry1], maxlen=PADDING_ON_S0, value=emptyIdx))[0]
        dataEntry2 = np.asarray(pad_sequences([dataEntry2], maxlen=PADDING_ON_S1, value=emptyIdx))[0]
        dataEntry3 = np.asarray(pad_sequences([dataEntry3], maxlen=PADDING_ON_B0, value=emptyIdx))[0]

        dataEntry1 = np.concatenate((dataEntry1, dataEntry2, dataEntry3), axis=0)
        return dataEntry1
