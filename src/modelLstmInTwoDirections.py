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

INPUT_LIST_NUM = 6

PADDING_ON_S0 = 5
PADDING_ON_S1 = 5
PADDING_ON_B0 = 2
INPUT_WORDS = PADDING_ON_S0 + PADDING_ON_S1 + PADDING_ON_B0


class NetworkLstmInTwoDirections(Network):
    def __init__(self, normalizer):
        sharedEmbedding = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                    weights=[normalizer.weightMatrix], trainable=True)
        sharedLSTM = LSTM(settings.LSTM_1_UNIT_NUM, name='sharedLSTMLayer')
        # S0-based left LSTM Module
        s0LeftInputLayer, s0LeftOutputLayer = self.createLSTMModule(PADDING_ON_S0, 's0Left', sharedEmbedding,
                                                                    sharedLSTM)
        # S0-based right LSTM Module
        s0RightInputLayer, s0RightOutputLayer = self.createLSTMModule(PADDING_ON_S0, 's0Right', sharedEmbedding,
                                                                      sharedLSTM, )
        # S1-based left LSTM Module
        s1LeftInputLayer, s1LeftOutputLayer = self.createLSTMModule(PADDING_ON_S1, 's1Left', sharedEmbedding,
                                                                    sharedLSTM, )
        # S1-based Right LSTM Module
        s1RightInputLayer, s1RightOutputLayer = self.createLSTMModule(PADDING_ON_S1, 's1Right', sharedEmbedding,
                                                                      sharedLSTM, )
        # Buffer-based Embedding Module
        bInputLayer = Input(shape=(PADDING_ON_B0,), name='bInputLayer')
        bOutputLayer = Flatten(name='bOutputLayer')(sharedEmbedding(bInputLayer))
        # Auxiliary feature vectors
        auxFeatureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='auxFeatureLayer')
        # Merge layer
        concLayer = keras.layers.concatenate([s0LeftOutputLayer, s0RightOutputLayer,
                                              s1LeftOutputLayer, s1RightOutputLayer,
                                              bOutputLayer, auxFeatureLayer])
        # MLP module
        mainOutputLayer = self.createMLPModule(concLayer)
        self.model = Model(inputs=[s0LeftInputLayer, s0RightInputLayer, s1LeftInputLayer,
                                   s1RightInputLayer, bInputLayer, auxFeatureLayer],
                           outputs=mainOutputLayer)
        super(NetworkLstmInTwoDirections, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]]), np.asarray([dataEntry[2]]), np.asarray(
            [dataEntry[3]]), np.asarray([dataEntry[4]]), np.asarray([dataEntry[5]])]
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=settings.NN_PREDICT_VERBOSE)
        return argmax(oneHotRep)


class NormalizerLstmInTwoDirections(Normalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(NormalizerLstmInTwoDirections, self).__init__(corpus, INPUT_LIST_NUM)

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
                np.asarray(list(reversed(pad_sequences([dataEntry1], maxlen=PADDING_ON_S0))))[0], \
                np.asarray(pad_sequences([dataEntry2], maxlen=PADDING_ON_S1, value=emptyIdx))[0], \
                np.asarray(list(reversed(pad_sequences([dataEntry2], maxlen=PADDING_ON_S1, value=emptyIdx))))[0], \
                np.asarray(pad_sequences([dataEntry3], maxlen=PADDING_ON_B0, value=emptyIdx))[0], \
                np.asarray(dataEntry4)]
