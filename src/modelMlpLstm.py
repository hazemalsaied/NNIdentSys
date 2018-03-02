import keras
import numpy as np
from keras.layers import Input, Embedding, Dense, LSTM, Dropout, GRU
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

from corpus import getTokens
from model import Network, Normalizer
from transitions import TransitionType
from vocabulary import empty
import settings

PREDICT_VERBOSE = 0

PADDING_ON_S0 = 4
PADDING_ON_S1 = 2
PADDING_ON_B0 = 2
INPUT_WORDS = PADDING_ON_S0 + PADDING_ON_S1 + PADDING_ON_B0




class NetworkMlpLstm(Network):
    def __init__(self, normalizer):
        # Auxiliary feature vectors
        auxFeatureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='aux_feature_Layer')
        wordLayer = Input(shape=(INPUT_WORDS,), name='word_layer')
        embeddingLayer = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                   weights=[normalizer.weightMatrix], trainable=True, name='emb_layer')(wordLayer)
        if settings.USE_GRU:
            if settings.STACKED:
                gru1 = GRU(settings.LSTM_1_UNIT_NUM, name='GRU_1', return_sequences=True)(embeddingLayer)
                gru2 = GRU(settings.LSTM_2_UNIT_NUM, name='GRU_2')(gru1)
            else:
                gru2 = GRU(settings.LSTM_1_UNIT_NUM, name='GRU_1')(embeddingLayer)
            concLayer = keras.layers.concatenate([gru2, auxFeatureLayer])
        else:
            if settings.STACKED:
                lstmLayer1 = LSTM(512, name='lstm_1', return_sequences=True)(embeddingLayer)
                lstmLayer2 = LSTM(512, name='lstm_2')(lstmLayer1)
            else:
                lstmLayer2 = LSTM(512, name='lstm_2')(embeddingLayer)
            concLayer = keras.layers.concatenate([lstmLayer2, auxFeatureLayer])
        # MLP module
        dense1Layer = Dense(settings.MLP_LAYER_1_UNIT_NUM, activation=self.getActiviation1())(concLayer)
        dropoutLayer = Dropout(0.2)(dense1Layer)
        dense2Layer = Dense(settings.MLP_LAYER_2_UNIT_NUM, activation=self.getActiviation1())(dropoutLayer)
        dropout2Layer = Dropout(0.2)(dense2Layer)
        dense3Layer = Dense(settings.MLP_LAYER_2_UNIT_NUM, activation=self.getActiviation2())(dropout2Layer)
        mainOutputLayer = Dense(len(TransitionType), activation='softmax', name='output_layer')(dense3Layer)
        self.model = Model(inputs=[wordLayer, auxFeatureLayer], outputs=mainOutputLayer)
        super(NetworkMlpLstm, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]])]
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=PREDICT_VERBOSE)
        return argmax(oneHotRep)


class NormalizerMlpLstm(Normalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(NormalizerMlpLstm, self).__init__(corpus, 2)

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

        dataEntry1 = np.asarray(pad_sequences([dataEntry1], maxlen=PADDING_ON_S0, value=emptyIdx))[0]
        dataEntry2 = np.asarray(pad_sequences([dataEntry2], maxlen=PADDING_ON_S1, value=emptyIdx))[0]
        dataEntry3 = np.asarray(pad_sequences([dataEntry3], maxlen=PADDING_ON_B0, value=emptyIdx))[0]

        dataEntry1 = np.concatenate((dataEntry1, dataEntry2, dataEntry3), axis=0)
        return [dataEntry1, np.asarray(dataEntry4)]
