import keras
import numpy as np
from keras.layers import Input, Embedding, Dense, LSTM, Dropout
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

import settings
from corpus import getTokens
from model import Normalizer, Network
from transitions import TransitionType
from vocabulary import empty

PHRASE_MAXIMUM_LENGHT = 100

USE_STACKED_LSTM = True

# 5 of S0, 5 of S1, 2 of the buffer
CONSIDERED_ELEMENTS = 12

INPUT_LIST_NUM = 6


class NetworkModel2(Network):
    def __init__(self, normalizer):
        sharedEmbedding = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                    weights=[normalizer.weightMatrix], trainable=True)
        sharedLSTM = LSTM(settings.LSTM_1_UNIT_NUM, name='sharedLSTMLayer1')
        sharedLSTM2 = LSTM(settings.LSTM_2_UNIT_NUM, return_sequences=True, name='sharedLSTMLayer2')
        # S0-based left LSTM Module
        w0LeftInputLayer, w0LeftOutputLayer = self.createLSTMModule('s0Left', sharedEmbedding, sharedLSTM, sharedLSTM2)
        # S0-based right LSTM Module
        wnRightInputLayer, wnRightOutputLayer = self.createLSTMModule('s0Right', sharedEmbedding, sharedLSTM,
                                                                      sharedLSTM2)
        # Auxiliary feature vectors
        auxFeatureLayer = Input(shape=(CONSIDERED_ELEMENTS, settings.LSTM_2_UNIT_NUM * PHRASE_MAXIMUM_LENGHT,),
                                name='auxFeatureLayer')
        # Merge layer
        concLayer = keras.layers.concatenate([w0LeftOutputLayer, wnRightOutputLayer])
        lambdalayer = keras.layers.Lambda([concLayer, auxFeatureLayer])
        # MLP module
        dense1Layer = Dense(settings.MLP_LAYER_1_UNIT_NUM, activation=self.getActiviation1())(lambdalayer)
        dropoutLayer = Dropout(0.2)(dense1Layer)
        dense2Layer = Dense(settings.MLP_LAYER_2_UNIT_NUM, activation=self.getActiviation2())(dropoutLayer)
        mainOutputLayer = Dense(len(TransitionType), activation='softmax', name='mainOutputLayer')(dense2Layer)
        self.model = Model(inputs=[w0LeftInputLayer, wnRightInputLayer, auxFeatureLayer],
                           outputs=mainOutputLayer)
        super(NetworkModel2, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]]), np.asarray([dataEntry[2]]), np.asarray(
            [dataEntry[3]]), np.asarray([dataEntry[4]]), np.asarray([dataEntry[5]])]
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=settings.NN_PREDICT_VERBOSE)
        return argmax(oneHotRep)


class NormalizerModel2(Normalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(NormalizerModel2, self).__init__(corpus, INPUT_LIST_NUM)

    def normalize(self, trans):
        dataEntry1, dataEntry2, dataEntry3, dataEntry4 = [], [], [], []
        if trans.configuration.stack:
            dataEntry1 = self.getIndices(getTokens(trans.configuration.stack[-1]))
            if len(trans.configuration.stack) > 1:
                dataEntry2 = self.getIndices(getTokens(trans.configuration.stack[-2]))
        if trans.configuration.buffer:
            dataEntry3 = self.getIndices(trans.configuration.buffer[:2])
        dataEntry4 = self.nnExtractor.vectorize(trans, self.corpus)
        emptyIdx = self.vocabulary.indices[empty]
        return [np.asarray(pad_sequences([dataEntry1], maxlen=5, value=emptyIdx), dtype='int64')[0], \
                np.asarray(list(reversed(pad_sequences([dataEntry1], maxlen=5, value=emptyIdx))), dtype='int64')[0], \
                np.asarray(pad_sequences([dataEntry2], maxlen=5, value=emptyIdx), dtype='int64')[0], \
                np.asarray(list(reversed(pad_sequences([dataEntry2], maxlen=5, value=emptyIdx))), dtype='int64')[0], \
                np.asarray(pad_sequences([dataEntry3], maxlen=2, value=emptyIdx), dtype='int64')[0], \
                np.asarray(dataEntry4, dtype='int32')]


from keras.models import Sequential
from keras.layers import Lambda

embeddingSize = 2
input1 = np.asarray([float(val) for val in np.random.uniform(low=-0.01, high=0.01, size=int(10 * embeddingSize))])
# input1 = [float(val) for val in np.random.uniform(low=-0.01, high=0.01, size=int(10 * embeddingSize))]
# print input1.shape
input2 = []
for i in range(3):
    input2.append(np.zeros((10 * embeddingSize)))
    # input2.append([0] * 10 * embeddingSize)
idx = 4
for i in range(3):
    input2[i][idx] = 1
    input2[i][idx + 1] = 1
    idx += 4
pass


def output_of_lambda(input_shape):
    return (embeddingSize * len(input2),)


def multiplay(x):
    input1 = x[0]
    input2 = x[1:]
    output = [0] * embeddingSize * len(input2)
    idx = 0
    for i in range(len(input2)):
        # startSliceIdx = input2[i].index(1)
        idxs = np.where(input2[i] == 1)[0]
        # listToInsert = multiplyLists(input1, input2[i])[idxs[0]:idxs[-1]+1]
        listToInsert = (input1 * input2[i])[idxs[0]:idxs[-1] + 1]
        listToInsertIdx = 0
        for ii in range(idx, idx + embeddingSize):
            output[ii] = listToInsert[listToInsertIdx]
            listToInsertIdx += 1
        idx = idx + embeddingSize
    return output


def multiplyLists(l1, l2):
    if len(l1) != len(l2):
        return
    output = [0] * len(l1)
    for i in range(len(output)):
        output[i] = l1[i] * l2[i]
    return output


inputs = [input1]
inputs.extend(input2)
multiplay(inputs)
model = Sequential()
model.add(Lambda(multiplay, output_shape=output_of_lambda))
