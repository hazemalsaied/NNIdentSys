import datetime

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Flatten, Embedding, Dropout
from keras.models import Model
from keras.utils import to_categorical
from numpy import argmax

import reports
from reports import *
from transitions import TransitionType


class Network:
    def __init__(self, normalizer):
        inputLayers, concLayers = [], []
        inputToken = Input((4,))
        inputLayers.append(inputToken)
        tokenEmb = configuration["model"]["embedding"]["tokenEmb"]
        tokenEmb = Embedding(len(normalizer.vocabulary.attachedTokens), tokenEmb)(inputToken)
        tokenFlatten = Flatten()(tokenEmb)
        concLayers.append(tokenFlatten)

        if configuration["model"]["embedding"]["usePos"]:
            inputPos = Input((4,))
            inputLayers.append(inputPos)
            posEmb = configuration["model"]["embedding"]["posEmb"]
            posEmb = Embedding(len(normalizer.vocabulary.attachedPos), posEmb)(inputPos)
            posFlatten = Flatten()(posEmb)
            concLayers.append(posFlatten)

        if configuration["features"]["active"]:
            featureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='Features')
            inputLayers.append(featureLayer)
            concLayers.append(featureLayer)
        if len(concLayers) > 1:
            conc = keras.layers.concatenate(concLayers)
        else:
            conc = concLayers[0]
        mlpConf = configuration["model"]["mlp"]
        lastLayer = conc
        dense1Conf = mlpConf["dense1"]
        if dense1Conf["active"]:
            dense1Layer = Dense(dense1Conf["unitNumber"], activation=dense1Conf["activation"])(conc)
            dropoutLayer = Dropout(0.2)(dense1Layer)
            lastLayer = dropoutLayer
        softmax = Dense(8, activation='softmax')(lastLayer)
        self.model = Model(inputs=inputLayers, outputs=softmax)
        sys.stdout.write('# Parameters = {0}\n'.format(self.model.count_params()))
        print self.model.summary()

    def predict(self, trans, normalizer):
        inputs = []
        tokenIdxs, posIdxs = normalizer.getAttachedIndices(trans)
        inputs.append(np.asarray([tokenIdxs]))
        if configuration["model"]["embedding"]["usePos"]:
            inputs.append(np.asarray([posIdxs]))
        if configuration["features"]["active"]:
            features = np.asarray(normalizer.nnExtractor.vectorize(trans))
            inputs.append(np.asarray([features]))
        oneHotRep = self.model.predict(inputs, batch_size=1,
                                       verbose=configuration["model"]["predict"]["verbose"])
        return argmax(oneHotRep)


def train(model, normalizer, corpus):
    trainConf = configuration["model"]["train"]
    model.compile(loss=trainConf["loss"], optimizer=trainConf["optimizer"], metrics=['accuracy'])
    bestWeightPath = reports.getBestWeightFilePath()
    callbacks = [
        ModelCheckpoint(bestWeightPath, monitor=trainConf["monitor"], verbose=1, save_best_only=True, mode='max')
    ] if bestWeightPath else []
    if trainConf["earlyStop"]:
        callbacks.append(EarlyStopping(
            monitor=trainConf["monitor"], min_delta=trainConf["minDelta"], patience=2, verbose=trainConf["verbose"]))
    time = datetime.datetime.now()
    labels, data = normalizer.generateLearningDataAttached(corpus)
    labels = to_categorical(labels, num_classes=len(TransitionType))
    model.fit(data, labels, validation_split=trainConf["validationSplit"],
              epochs=trainConf["epochs"],
              batch_size=trainConf["batchSize"],
              verbose=trainConf["verbose"],
              callbacks=callbacks)
    sys.stdout.write('# Training time = {0}\n'.format(datetime.datetime.now() - time))
