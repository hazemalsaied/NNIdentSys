import datetime

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, GRU
from keras.models import Model
from keras.utils import to_categorical
from numpy import argmax

import reports
from reports import *
from transitions import TransitionType


class Network:
    def __init__(self, normalizer):
        rnnConf = configuration["model"]["rnn"]

        embConf = configuration["model"]["embedding"]
        inputLayers, concLayers = [], []
        inputToken = Input((4,))
        inputLayers.append(inputToken)
        tokenEmb = configuration["model"]["embedding"]["tokenEmb"]
        if embConf["initialisation"]["active"] and embConf["initialisation"]["token"]:
            sys.stdout.write('# Token weight matrix used')
            tokenWeights = [normalizer.tokenWeightMatrix]
            tokenEmb = Embedding(len(tokenWeights[0]), tokenEmb, weights=tokenWeights,
                                 trainable=True)(inputToken)
        else:
            tokenEmb = Embedding(len(normalizer.vocabulary.attachedTokens), tokenEmb)(inputToken)
        if rnnConf["active"]:
            tokenRnn = GRU(rnnConf["rnn1"]["unitNumber"])(tokenEmb)#, return_sequences=True)
            concLayers.append(tokenRnn)
        else:
            tokenFlatten = Flatten()(tokenEmb)
            concLayers.append(tokenFlatten)

        if embConf["usePos"]:
            inputPos = Input((4,))
            inputLayers.append(inputPos)
            posEmb = configuration["model"]["embedding"]["posEmb"]

            if embConf["initialisation"]["active"] and embConf["initialisation"]["pos"]:
                sys.stdout.write('# POS weight matrix used')
                weights = [normalizer.posWeightMatrix]
                posEmb = Embedding(len(normalizer.vocabulary.attachedPos), posEmb, weights=weights, trainable=True)(
                    inputPos)
            else:
                posEmb = Embedding(len(normalizer.vocabulary.attachedPos), posEmb)(inputPos)
            if rnnConf["active"]:
                posRnn = GRU(rnnConf["rnn1"]["posUnitNumber"])(posEmb) #, return_sequences=True)
                concLayers.append(posRnn)
            else:
                posFlatten = Flatten()(posEmb)
                concLayers.append(posFlatten)

        if configuration["features"]["active"]:
            featureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='Features')
            inputLayers.append(featureLayer)
            concLayers.append(featureLayer)
        conc = keras.layers.concatenate(concLayers) if len(concLayers) > 1 else concLayers[0]
        mlpConf = configuration["model"]["mlp"]
        lastLayer = conc
        dense1Conf = mlpConf["dense1"]
        if dense1Conf["active"]:
            dense1Layer = Dense(dense1Conf["unitNumber"], activation=dense1Conf["activation"])(conc)
            lastLayer = Dropout(0.2)(dense1Layer)
        dense2Conf = mlpConf["dense2"]
        if dense2Conf["active"]:
            dense2Layer = Dense(dense2Conf["unitNumber"], activation=dense2Conf["activation"])(lastLayer)
            lastLayer = Dropout(0.2)(dense2Layer)
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
