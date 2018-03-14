import keras
import numpy as np
from keras.layers import Input, Embedding, Dense, LSTM, GRU, TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

from corpus import getTokens
from model import AbstractNetwork, AbstractNormalizer
from config import configuration
from transitions import TransitionType
from vocabulary import empty

USE_STACKED_LSTM = False

INPUT_LIST_NUM = 6

INPUT_WORDS = configuration["model"]["embedding"]["s0Padding"] + configuration["model"]["embedding"]["s1Padding"] + \
              configuration["model"]["embedding"]["bPadding"]


class Network(AbstractNetwork):
    def __init__(self, normalizer):

        # Auxiliary feature vectors
        auxFeatureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='aux_feature')
        dense1 = Dense(configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"], name='dense_1')(
            auxFeatureLayer)
        dense2 = Dense(configuration["model"]["topology"]["mlp"]["dense2"]["unitNumber"], name='dense_2')(dense1)

        wordLayer = Input(shape=(INPUT_WORDS,), name='words')
        embeddingLayer = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                   weights=[normalizer.weightMatrix], trainable=True, name='emb')(wordLayer)
        if configuration["model"]["topology"]["rnn"]["gru"]:
            if configuration["model"]["topology"]["rnn"]["stacked"]:
                iLay1 = GRU(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], dropout=.2, name='i_gru_1',
                            return_sequences=True)(embeddingLayer)
                iLay2 = GRU(configuration["model"]["topology"]["rnn"]["rnn2"]["unitNumber"], dropout=.2,
                            name='i_gru_2')(iLay1)
            else:
                iLay2 = GRU(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], dropout=.2,
                            name='i_gru_1')(embeddingLayer)
        else:
            if configuration["model"]["topology"]["rnn"]["stacked"]:
                iLay1 = LSTM(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], dropout=.2,
                             name='ilstm_1', return_sequences=True)(
                    embeddingLayer)
                iLay2 = LSTM(configuration["model"]["topology"]["rnn"]["rnn2"]["unitNumber"], dropout=.2,
                             name='ilstm_2')(iLay1)
            else:
                iLay2 = LSTM(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], dropout=.2,
                             name='ilstm_2')(embeddingLayer)

        concLayer = TimeDistributed(keras.layers.concatenate([iLay2, dense2], name='Conc_1'), input_shape=(100, 1024))

        if configuration["model"]["topology"]["rnn"]["gru"]:
            if configuration["model"]["topology"]["rnn"]["stacked"]:
                lay1 = GRU(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], dropout=.2, name='gru_1',
                           return_sequences=True)(concLayer)
                lay2 = GRU(configuration["model"]["topology"]["rnn"]["rnn2"]["unitNumber"], dropout=.2, name='gru_2')(
                    lay1)
            else:
                lay2 = GRU(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], dropout=.2, name='gru_1')(
                    concLayer)
        else:
            if configuration["model"]["topology"]["rnn"]["stacked"]:
                lay1 = LSTM(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], dropout=.2, name='lstm_1',
                            return_sequences=True)(concLayer)
                lay2 = LSTM(configuration["model"]["topology"]["rnn"]["rnn2"]["unitNumber"], dropout=.2, name='lstm_2')(
                    lay1)
            else:
                lay2 = LSTM(configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"], dropout=.2, name='lstm_1')(
                    concLayer)
        mainOutputLayer = Dense(len(TransitionType), activation='softmax', name='output')(lay2)
        self.model = Model(inputs=[wordLayer, auxFeatureLayer], outputs=mainOutputLayer)
        super(Network, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]]), np.asarray([dataEntry[2]]), np.asarray(
            [dataEntry[3]]), np.asarray([dataEntry[4]]), np.asarray([dataEntry[5]])]
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=configuration["model"]["predict"]["verbose"])
        return argmax(oneHotRep)


class Normalizer(AbstractNormalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(Normalizer, self).__init__(corpus, INPUT_LIST_NUM)

    def normalize(self, trans):
        dataEntry1, dataEntry2, dataEntry3, dataEntry4 = [], [], [], []
        if trans.configuration.stack:
            dataEntry1 = self.getIndices(getTokens(trans.configuration.stack[-1]))
            if len(trans.configuration.stack) > 1:
                dataEntry2 = self.getIndices(getTokens(trans.configuration.stack[-2]))
        if trans.configuration.buffer:
            dataEntry3 = self.getIndices(trans.configuration.buffer[:2])
        dataEntry4 = np.asarray(self.nnExtractor.vectorize(trans))
        emptyIdx = self.vocabulary.indices[empty]

        dataEntry1 = np.asarray(
            pad_sequences([dataEntry1], maxlen=configuration["model"]["embedding"]["s0Padding"], value=emptyIdx))[0]
        dataEntry2 = np.asarray(
            pad_sequences([dataEntry2], maxlen=configuration["model"]["embedding"]["s1Padding"], value=emptyIdx))[0]
        dataEntry3 = \
        np.asarray(pad_sequences([dataEntry3], maxlen=configuration["model"]["embedding"]["bPadding"], value=emptyIdx))[
            0]

        dataEntry1 = np.concatenate((dataEntry1, dataEntry2, dataEntry3), axis=0)
        return [dataEntry1, dataEntry4]
