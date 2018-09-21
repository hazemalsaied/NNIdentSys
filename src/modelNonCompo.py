from collections import Counter

import keras
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Flatten, Embedding, Dropout
from keras.models import Model
from keras.utils import to_categorical

import reports
import sampling
from reports import *


class Network:
    def __init__(self, normalizer):
        sys.stdout.write('Deep model(Non compositional)\n')
        embConf = configuration["model"]["embedding"]
        inputLayers, concLayers = [], []
        inputToken = Input((configuration['model']['inputItems'],))
        inputLayers.append(inputToken)
        tokenEmb = configuration["model"]["embedding"]["tokenEmb"]
        if embConf["initialisation"]["active"] and not embConf["initialisation"]["modifiable"]:
            sys.stdout.write('# Token weight matrix used')
            tokenWeights = [normalizer.tokenWeightMatrix]
            tokenEmb = Embedding(len(tokenWeights[0]), tokenEmb, weights=tokenWeights,
                                 trainable=False)(inputToken)
        elif embConf["initialisation"]["active"] and embConf["initialisation"]["token"]:
            sys.stdout.write('# Token weight matrix used')
            tokenWeights = [normalizer.tokenWeightMatrix]
            tokenEmb = Embedding(len(tokenWeights[0]), tokenEmb, weights=tokenWeights,
                                 trainable=True)(inputToken)
        else:
            tokenEmb = Embedding(len(normalizer.vocabulary.tokenIndices), tokenEmb)(inputToken)

        tokenFlatten = Flatten()(tokenEmb)
        concLayers.append(tokenFlatten)

        inputPos = Input((configuration['model']['inputItems'],))
        inputLayers.append(inputPos)
        posEmb = configuration["model"]["embedding"]["posEmb"]

        if embConf["initialisation"]["active"] and embConf["initialisation"]["pos"]:
            sys.stdout.write('# POS weight matrix used')
            weights = [normalizer.posWeightMatrix]
            posEmb = Embedding(len(normalizer.vocabulary.posIndices), posEmb, weights=weights, trainable=True)(
                inputPos)
        else:
            posEmb = Embedding(len(normalizer.vocabulary.posIndices), posEmb)(inputPos)
        posFlatten = Flatten()(posEmb)
        concLayers.append(posFlatten)

        conc = keras.layers.concatenate(concLayers) if len(concLayers) > 1 else concLayers[0]
        mlpConf = configuration["model"]["mlp"]
        lastLayer = conc
        dense1Conf = mlpConf["dense1"]
        if dense1Conf["active"]:
            dense1Layer = Dense(dense1Conf["unitNumber"], activation=dense1Conf["activation"])(conc)
            lastLayer = Dropout(dense1Conf["dropout"])(dense1Layer)
            dense2Conf = mlpConf["dense2"]
            if dense2Conf["active"]:
                dense2Layer = Dense(dense2Conf["unitNumber"], activation=dense2Conf["activation"])(lastLayer)
                lastLayer = Dropout(dense2Conf["dropout"])(dense2Layer)
        softmaxLayer = Dense(8, activation='softmax')(lastLayer)
        self.model = Model(inputs=inputLayers, outputs=softmaxLayer)
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
        return oneHotRep[0]


def train(model, normalizer, corpus):
    trainConf = configuration["model"]["train"]
    labels, data = normalizer.generateLearningDataAttached(corpus)
    sys.stdout.write(reports.seperator + reports.tabs + 'Sampling' + reports.doubleSep)
    normalizer.inputListDimension = 2
    if configuration["sampling"]["focused"]:
        data, labels = sampling.overSampleImporTrans(data, labels, corpus, normalizer)
    labels, data = sampling.overSample(labels, data)
    if configuration['model']['train']['earlyStop']:
        # To make sure that we will get a random validation dataset
        labels, data = sampling.shuffleTwoArrayInParallel(labels, data)
    lblDistribution = Counter(labels)
    sys.stdout.write(tabs + '{0} Labels in train : {1}\n'.format(len(lblDistribution), lblDistribution))
    valDistribution = Counter(labels[int(len(labels) * (1 - trainConf["validationSplit"])):])
    sys.stdout.write(tabs + '{0} Labels in valid : {1}\n'.format(len(valDistribution), valDistribution))
    classWeightDic = sampling.getClassWeights(labels)
    sampleWeights = sampling.getSampleWeightArray(labels, classWeightDic)
    labels = to_categorical(labels, num_classes=8)
    model.compile(loss=trainConf["loss"], optimizer=getOptimizer(), metrics=['accuracy'])
    history = model.fit(data, labels,
                        validation_split=trainConf["validationSplit"],
                        epochs=trainConf["epochs"],
                        batch_size=trainConf["batchSize"],
                        verbose=2,
                        callbacks=getCallBacks(),
                        sample_weight=sampleWeights)
    if trainConf["verbose"]:
        sys.stdout.write('Epoch Losses= ' + str(history.history['loss']))
    trainValidationData(model, normalizer, data, labels, classWeightDic, history)


def getOptimizer():
    sys.stdout.write(reports.seperator + reports.tabs +
                     'Optimizer : Adagrad,  learning rate = {0}'.format(configuration["model"]["train"]["lr"])
                     + reports.seperator)
    return optimizers.Adagrad(lr=configuration["model"]["train"]["lr"], epsilon=None, decay=0.0)


def trainValidationData(model, normalizer, data, labels, classWeightDic, history):
    trainConf = configuration["model"]["train"]
    data, labels = getValidationData(normalizer, data, labels)
    validationLabelsAsInt = [np.where(r == 1)[0][0] for r in labels]
    sampleWeights = sampling.getSampleWeightArray(validationLabelsAsInt, classWeightDic)
    history = model.fit(data, labels,
                        epochs=len(history.epoch),
                        batch_size=trainConf["batchSize"],
                        verbose=0,
                        sample_weight=sampleWeights)
    return history


def getValidationData(normalizer, data, labels):
    trainConf = configuration["model"]["train"]
    validationData = []
    if normalizer.inputListDimension and normalizer.inputListDimension != 1:
        for dataTensor in data:
            validationData.append(dataTensor[int(len(dataTensor) * (1 - trainConf["validationSplit"])):])
    else:
        validationData = data[int(len(data) * (1 - trainConf["validationSplit"])):]
    validationLabel = labels[int(len(labels) * (1 - trainConf["validationSplit"])):]
    return validationData, validationLabel


def getCallBacks():
    trainConf = configuration["model"]["train"]
    bestWeightPath = reports.getBestWeightFilePath()
    callbacks = [
        ModelCheckpoint(bestWeightPath, monitor=trainConf["monitor"], verbose=1, save_best_only=True, mode='max')
    ] if bestWeightPath else []
    # callbacks.append(plot_losses)
    if trainConf["earlyStop"]:
        es = EarlyStopping(monitor=trainConf["monitor"],
                           min_delta=trainConf["minDelta"],
                           patience=2,
                           verbose=trainConf["verbose"])
        callbacks.append(es)
    return callbacks
