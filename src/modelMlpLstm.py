import keras
import numpy as np
from keras.layers import Input, Embedding, LSTM, GRU
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

from config import configuration
from corpus import getTokens
from model import AbstractNetwork, AbstractNormalizer
from vocabulary import empty

PREDICT_VERBOSE = 0

INPUT_WORDS = configuration["model"]["embedding"]["s0Padding"] + configuration["model"]["embedding"]["s1Padding"] + configuration["model"]["embedding"]["bPadding"]


class Network(AbstractNetwork):
    def __init__(self, normalizer):
        # Auxiliary feature vectors
        auxFeatureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='aux_feature_Layer')
        wordLayer = Input(shape=(INPUT_WORDS,), name='word_layer')
        embeddingLayer = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                   weights=[normalizer.weightMatrix], trainable=True, name='emb_layer')(wordLayer)
        if configuration["model"]["topology"]["rnn"]["gru"]:
            if configuration["model"]["topology"]["rnn"]["stacked"]:
                gru1 = GRU(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], name='GRU_1', return_sequences=True)(embeddingLayer)
                gru2 = GRU(configuration["model"]["topology"]["rnn"]["rnn2"]["unitNumber"], name='GRU_2')(gru1)
            else:
                gru2 = GRU(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], name='GRU_1')(embeddingLayer)
            concLayer = keras.layers.concatenate([gru2, auxFeatureLayer])
        else:
            if configuration["model"]["topology"]["rnn"]["stacked"]:
                lstmLayer1 = LSTM(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], name='lstm_1', return_sequences=True)(embeddingLayer)
                lstmLayer2 = LSTM(configuration["model"]["topology"]["rnn"]["rnn2"]["unitNumber"], name='lstm_2')(lstmLayer1)
            else:
                lstmLayer2 = LSTM(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], name='lstm_2')(embeddingLayer)
            concLayer = keras.layers.concatenate([lstmLayer2, auxFeatureLayer])
        # MLP module
        mainOutputLayer = self.createMLPModule(concLayer)
        self.model = Model(inputs=[wordLayer, auxFeatureLayer], outputs=mainOutputLayer)
        super(Network, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]])]
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=PREDICT_VERBOSE)
        return argmax(oneHotRep)


class Normalizer(AbstractNormalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(Normalizer, self).__init__(corpus, 2)

    def normalize(self, trans):
        dataEntry1, dataEntry2, dataEntry3, dataEntry4 = [], [], [], []
        if trans.configuration.stack:
            dataEntry1 = self.getIndices(getTokens(trans.configuration.stack[-1]))
            if len(trans.configuration.stack) > 1:
                dataEntry2 = self.getIndices(getTokens(trans.configuration.stack[-2]))
        if trans.configuration.buffer:
            dataEntry3 = self.getIndices(trans.configuration.buffer[:2])
        dataEntry4 = self.nnExtractor.vectorize(trans)
        emptyIdx = self.vocabulary.indices[empty]

        dataEntry1 = np.asarray(pad_sequences([dataEntry1], maxlen=configuration["model"]["embedding"]["s0Padding"], value=emptyIdx))[0]
        dataEntry2 = np.asarray(pad_sequences([dataEntry2], maxlen=configuration["model"]["embedding"]["s1Padding"], value=emptyIdx))[0]
        dataEntry3 = np.asarray(pad_sequences([dataEntry3], maxlen=configuration["model"]["embedding"]["bPadding"], value=emptyIdx))[0]

        dataEntry1 = np.concatenate((dataEntry1, dataEntry2, dataEntry3), axis=0)
        return [dataEntry1, np.asarray(dataEntry4)]
