import datetime

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Flatten, Embedding
from keras.models import Model
from keras.utils import to_categorical
from numpy import argmax

import reports
from reports import *
from transitions import TransitionType


class Network:
    def __init__(self, normalizer):
        inputToken = Input((4,))
        inputPos = Input((4,))
        tokenEmb = Embedding(len(normalizer.vocabulary.attachedTokens), 200)(inputToken)
        posEmb = Embedding(len(normalizer.vocabulary.attachedPos), 50)(inputPos)
        tokenFlatten = Flatten()(tokenEmb)
        posFlatten = Flatten()(posEmb)
        conc = keras.layers.concatenate([tokenFlatten, posFlatten])
        dense1 = Dense(256, activation='relu')(conc)
        dense2 = Dense(8, activation='softmax')(dense1)
        self.model = Model(inputs=[inputToken, inputPos], outputs=dense2)
        print self.model.summary()

    def predict(self, trans, normalizer):
        tokenIdxs, posIdxs = normalizer.getAttachedIndices(trans)
        oneHotRep = self.model.predict([np.asarray([tokenIdxs]), np.asarray([posIdxs])], batch_size=1,
                                       verbose=configuration["model"]["predict"]["verbose"])
        return argmax(oneHotRep)


def train(model, normalizer, corpus):
    trainConf = configuration["model"]["train"]
    model.compile(loss=trainConf["loss"], optimizer=trainConf["optimizer"], metrics=['accuracy'])
    bestWeightPath = reports.getBestWeightFilePath()
    callbacks = [
        ModelCheckpoint(bestWeightPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    ] if bestWeightPath else []
    if trainConf["earlyStop"]:
        callbacks.append(EarlyStopping(
            monitor='val_acc', min_delta=.5, patience=2, verbose=trainConf["verbose"]))
    time = datetime.datetime.now()
    logging.warn('Training started!')
    labels, data1, data2 = normalizer.generateLearningDataAttached(corpus)
    labels = to_categorical(labels, num_classes=len(TransitionType))
    model.fit([data1, data2], labels, validation_split=0.2,
              epochs=trainConf["epochs"],
              batch_size=trainConf["batchSize"],
              verbose=trainConf["verbose"],
              callbacks=callbacks)
    logging.warn('Training has taken: {0}!'.format(datetime.datetime.now() - time))
