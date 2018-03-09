import keras
import numpy as np
from keras.layers import Input, Embedding, Flatten, LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

import settings
from corpus import getTokens
from model import Network, Normalizer
from vocabulary import empty

USE_STACKED_LSTM = False

INPUT_LIST_NUM = 4

PADDING_ON_S0 = 4
PADDING_ON_S1 = 4
PADDING_ON_B0 = 2
INPUT_WORDS = PADDING_ON_S0 + PADDING_ON_S1 + PADDING_ON_B0


class NetworkMlpLstm2(Network):
    def __init__(self, normalizer):
        sharedEmbedding = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                    weights=[normalizer.weightMatrix], trainable=True)
        sharedLSTM = LSTM(settings.LSTM_1_UNIT_NUM, name='sharedLSTMLayer')
        # S0-based left LSTM Module
        s0LeftInputLayer, s0LeftOutputLayer = self.createLSTMModule(PADDING_ON_S0, 's0Left', sharedEmbedding,
                                                                    sharedLSTM)
        # S1-based left LSTM Module
        s1LeftInputLayer, s1LeftOutputLayer = self.createLSTMModule(PADDING_ON_S1, 's1Left', sharedEmbedding,
                                                                    sharedLSTM, )
        # Buffer-based Embedding Module
        bInputLayer = Input(shape=(PADDING_ON_B0,), name='bInputLayer')
        bOutputLayer = Flatten(name='bOutputLayer')(sharedEmbedding(bInputLayer))
        # Auxiliary feature vectors
        auxFeatureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='auxFeatureLayer')
        # Merge layer
        concLayer = keras.layers.concatenate([s0LeftOutputLayer, s1LeftOutputLayer,
                                              bOutputLayer, auxFeatureLayer])
        # MLP module
        mainOutputLayer = self.createMLPModule(concLayer)
        self.model = Model(inputs=[s0LeftInputLayer, s1LeftInputLayer, bInputLayer, auxFeatureLayer],
                           outputs=mainOutputLayer)
        super(NetworkMlpLstm2, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]]), np.asarray([dataEntry[2]]), np.asarray(
            [dataEntry[3]])]
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=settings.NN_PREDICT_VERBOSE)
        return argmax(oneHotRep)


class NormalizerMlpLstm2(Normalizer):
    """
        Responsible for transforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(NormalizerMlpLstm2, self).__init__(corpus, INPUT_LIST_NUM)

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
        return [np.asarray(pad_sequences([dataEntry1], maxlen=PADDING_ON_S0, value=emptyIdx))[0], \
                np.asarray(pad_sequences([dataEntry2], maxlen=PADDING_ON_S1, value=emptyIdx))[0], \
                np.asarray(pad_sequences([dataEntry3], maxlen=PADDING_ON_B0, value=emptyIdx))[0], \
                np.asarray(dataEntry4)]
