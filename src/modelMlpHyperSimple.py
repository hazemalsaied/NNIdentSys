import numpy as np
from keras.layers import Input, Embedding, Flatten
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

from config import configuration
from corpus import getTokens
from model import AbstractNormalizer, AbstractNetwork
from vocabulary import empty

INPUT_WORDS = configuration["model"]["embedding"]["s0Padding"] + configuration["model"]["embedding"]["s1Padding"] + configuration["model"]["embedding"]["bPadding"]


class Network(AbstractNetwork):
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

        super(Network, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        dataEntry = np.reshape(dataEntry, (1, len(dataEntry)))
        oneHotRep = self.model.predict(dataEntry, batch_size=1, verbose=configuration["model"]["predict"]["verbose"])
        # oneHotRep = self.model.predict(np.asarray([dataEntry]), batch_size=1, verbose=PREDICT_VERBOSE)
        return argmax(oneHotRep)


class Normalizer(AbstractNormalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(Normalizer, self).__init__(corpus, 1, False)

    def normalize(self, trans):
        dataEntry1, dataEntry2, dataEntry3 = [], [], []
        if trans.configuration.stack:
            dataEntry1 = self.getIndices(getTokens(trans.configuration.stack[-1]))
            if len(trans.configuration.stack) > 1:
                dataEntry2 = self.getIndices(getTokens(trans.configuration.stack[-2]))
        if trans.configuration.buffer:
            dataEntry3 = self.getIndices(trans.configuration.buffer[:2])
        emptyIdx = self.vocabulary.indices[empty]

        dataEntry1 = np.asarray(pad_sequences([dataEntry1], maxlen=configuration["model"]["embedding"]["s0Padding"], value=emptyIdx))[0]
        dataEntry2 = np.asarray(pad_sequences([dataEntry2], maxlen=configuration["model"]["embedding"]["s1Padding"], value=emptyIdx))[0]
        dataEntry3 = np.asarray(pad_sequences([dataEntry3], maxlen=configuration["model"]["embedding"]["bPadding"], value=emptyIdx))[0]

        dataEntry1 = np.concatenate((dataEntry1, dataEntry2, dataEntry3), axis=0)
        return dataEntry1
