import keras
import numpy as np
from keras.layers import Input, Embedding, Dense, LSTM, GRU, TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

import settings
from corpus import getTokens
from model import Network, Normalizer
from transitions import TransitionType
from vocabulary import empty

USE_STACKED_LSTM = False

INPUT_LIST_NUM = 6

INPUT_WORDS = settings.PADDING_ON_S0 + settings.PADDING_ON_S1 + settings.PADDING_ON_B0


class NetworkConfTrans(Network):
    def __init__(self, normalizer):

        # Auxiliary feature vectors
        auxFeatureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='aux_feature')
        dense1 = Dense(settings.MLP_LAYER_1_UNIT_NUM, name='dense_1')(auxFeatureLayer)
        dense2 = Dense(settings.MLP_LAYER_2_UNIT_NUM, name='dense_2')(dense1)

        wordLayer = Input(shape=(INPUT_WORDS,), name='words')
        embeddingLayer = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                   weights=[normalizer.weightMatrix], trainable=True, name='emb')(wordLayer)
        if settings.USE_GRU:
            if settings.STACKED:
                iLay1 = GRU(settings.LSTM_1_UNIT_NUM, dropout=.2, name='i_gru_1', return_sequences=True)(embeddingLayer)
                iLay2 = GRU(settings.LSTM_2_UNIT_NUM, dropout=.2, name='i_gru_2')(iLay1)
            else:
                iLay2 = GRU(settings.LSTM_1_UNIT_NUM, dropout=.2, name='i_gru_1')(embeddingLayer)
        else:
            if settings.STACKED:
                iLay1 = LSTM(settings.LSTM_1_UNIT_NUM, dropout=.2, name='ilstm_1', return_sequences=True)(
                    embeddingLayer)
                iLay2 = LSTM(settings.LSTM_2_UNIT_NUM, dropout=.2, name='ilstm_2')(iLay1)
            else:
                iLay2 = LSTM(settings.LSTM_1_UNIT_NUM, dropout=.2, name='ilstm_2')(embeddingLayer)

        concLayer = TimeDistributed(keras.layers.concatenate([iLay2, dense2], name='Conc_1'), input_shape=(100, 1024))

        if settings.USE_GRU:
            if settings.STACKED:
                lay1 = GRU(settings.LSTM_1_UNIT_NUM, dropout=.2, name='gru_1', return_sequences=True)(concLayer)
                lay2 = GRU(settings.LSTM_2_UNIT_NUM, dropout=.2, name='gru_2')(lay1)
            else:
                lay2 = GRU(settings.LSTM_1_UNIT_NUM, dropout=.2, name='gru_1')(concLayer)
        else:
            if settings.STACKED:
                lay1 = LSTM(settings.LSTM_1_UNIT_NUM, dropout=.2, name='lstm_1', return_sequences=True)(concLayer)
                lay2 = LSTM(settings.LSTM_2_UNIT_NUM, dropout=.2, name='lstm_2')(lay1)
            else:
                lay2 = LSTM(settings.LSTM_1_UNIT_NUM, dropout=.2, name='lstm_1')(concLayer)
        mainOutputLayer = Dense(len(TransitionType), activation='softmax', name='output')(lay2)
        self.model = Model(inputs=[wordLayer, auxFeatureLayer], outputs=mainOutputLayer)
        super(NetworkConfTrans, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]]), np.asarray([dataEntry[2]]), np.asarray(
            [dataEntry[3]]), np.asarray([dataEntry[4]]), np.asarray([dataEntry[5]])]
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=settings.NN_PREDICT_VERBOSE)
        return argmax(oneHotRep)


class NormalizerConfTrans(Normalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(NormalizerConfTrans, self).__init__(corpus, INPUT_LIST_NUM)

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

        dataEntry1 = np.asarray(pad_sequences([dataEntry1], maxlen=settings.PADDING_ON_S0, value=emptyIdx))[0]
        dataEntry2 = np.asarray(pad_sequences([dataEntry2], maxlen=settings.PADDING_ON_S1, value=emptyIdx))[0]
        dataEntry3 = np.asarray(pad_sequences([dataEntry3], maxlen=settings.PADDING_ON_B0, value=emptyIdx))[0]

        dataEntry1 = np.concatenate((dataEntry1, dataEntry2, dataEntry3), axis=0)
        return [dataEntry1, dataEntry4]
