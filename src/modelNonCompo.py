import datetime

import keras
import numpy as np
import sklearn.utils
from IPython.display import clear_output
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, GRU
from keras.models import Model
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import argmax

import reports
from reports import *
from transitions import TransitionType


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batchStep = configuration["model"]["train"]["visualisation"]["batchStep"]
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        # self.fig = plt.figure()
        self.batchNum = 0
        self.batchLosses = []
        self.batchValLosses = []
        self.batchAcc = []
        self.batchValAcc = []
        self.logs = []

    # def on_epoch_end(self, epoch, logs={}):
    #     self.logs.append(logs)
    #     self.x.append(self.i)
    #     self.losses.append(logs.get('loss'))
    #     self.val_losses.append(logs.get('val_loss'))
    #     self.acc.append(logs.get('acc'))
    #     self.val_acc.append(logs.get('val_acc'))
    #
    #
    #     self.i += 1
    #     f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    #
    #     clear_output(wait=True)
    #
    #     ax1.set_yscale('log')
    #     ax1.plot(self.x, self.losses, label="loss")
    #     ax1.plot(self.x, self.val_losses, label="val_loss")
    #     ax1.legend()
    #
    #     ax2.plot(self.x, self.acc, label="accuracy")
    #     ax2.plot(self.x, self.val_acc, label="validation accuracy")
    #     ax2.legend()
    #
    #     plt.show()

    # def on_batch_end(self, batch, logs=None):
    #     self.batchNum += 1
    #     if self.batchNum % self.batchStep == 0:
    #         self.batchLosses.append(logs.get('loss'))
    #         # self.batchValLosses.append(logs.get('val_loss'))
    #         self.batchAcc.append(logs.get('acc'))
    #         # self.batchValAcc.append(logs.get('val_acc'))

    def on_train_end(self, logs=None):
        if not configuration["evaluation"]["cluster"]:
            f, ax1 = plt.subplots()
            clear_output(wait=True)
            ax1.set_yscale('log')
            ax1.plot([(x + 1) * self.batchStep for x in range(len(self.batchAcc))], self.batchLosses, label="loss ")
            ax1.plot([(x + 1) * self.batchStep for x in range(len(self.batchAcc))], self.batchAcc, label="Acc ")
            # ax1.legend()

            # ax2.plot(self.x, self.acc, label="accuracy")
            # ax1.plot(range(len(self.batchAcc)), self.batchAcc, label="loss (batch)")
            # ax2.legend()
            plt.show()


plot_losses = PlotLosses()


class Network:
    def __init__(self, normalizer):
        sys.stdout.write('Deep model(Non compositional)\n')
        rnnConf = configuration["model"]["rnn"]

        embConf = configuration["model"]["embedding"]
        inputLayers, concLayers = [], []
        inputToken = Input((4,))
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
            tokenEmb = Embedding(len(normalizer.vocabulary.attachedTokens), tokenEmb)(inputToken)
        if rnnConf["active"]:
            tokenRnn = GRU(rnnConf["rnn1"]["unitNumber"])(tokenEmb)  # , return_sequences=True)
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
                posRnn = GRU(rnnConf["rnn1"]["posUnitNumber"])(posEmb)  # , return_sequences=True)
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
        # sys.stdout.write('# Parameters = {0}\n'.format(self.model.count_params()))
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
    time = datetime.datetime.now()
    trainConf = configuration["model"]["train"]
    labels, data = normalizer.generateLearningDataAttached(corpus)
    classWeightDic = getClassWeights(labels)
    sampleWeights = getSampleWeightArray(labels, classWeightDic)
    labels = to_categorical(labels, num_classes=len(TransitionType))
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
    sys.stdout.write(reports.doubleSep + reports.tabs + 'Training time : {0}'.format(datetime.datetime.now() - time)
                     + reports.doubleSep)


def getOptimizer():
    trainConf = configuration["model"]["train"]
    sys.stdout.write(reports.seperator + reports.tabs +
                     'Optimizer : {0},  learning rate = {1}'.format(trainConf["optimizer"], trainConf["lr"])
                     + reports.seperator)
    lr = trainConf["lr"]
    if trainConf["optimizer"] == 'sgd':
        return optimizers.SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False)
    if trainConf["optimizer"] == 'adam':
        return optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if trainConf["optimizer"] == 'rmsprop':
        return optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
    if trainConf["optimizer"] == 'adagrad':
        return optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
    if trainConf["optimizer"] == 'adadelta':
        return optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)  # lr = 1.
    if trainConf["optimizer"] == 'adamax':
        return optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)  # lr = 0.002
    if trainConf["optimizer"] == 'nadam':
        return optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
                                schedule_decay=0.004)  # lr = 0.002
    assert 'No optimizer found for training!'


def trainValidationData(model, normalizer, data, labels, classWeightDic, history):
    # sys.stdout.write('Training over validation data\n')
    trainConf = configuration["model"]["train"]
    data, labels = getValidationData(normalizer, data, labels)
    validationLabelsAsInt = [np.where(r == 1)[0][0] for r in labels]
    sampleWeights = getSampleWeightArray(validationLabelsAsInt, classWeightDic)
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
    callbacks.append(plot_losses)
    if trainConf["earlyStop"]:
        callbacks.append(EarlyStopping(monitor=trainConf["monitor"],
                                       min_delta=trainConf["minDelta"],
                                       patience=2,
                                       verbose=trainConf["verbose"]))
    return callbacks


def getSampleWeightArray(labels, classWeightDic):
    if not configuration["sampling"]["sampleWeight"]:
        return None
    sampleWeights = []
    for l in labels:
        sampleWeights.append(classWeightDic[l])
    return np.asarray(sampleWeights)


def getClassWeights(labels):
    if not configuration["model"]["train"]["manipulateClassWeights"]:
        return {}
    classes = np.unique(labels)
    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes, labels)
    res = dict()
    for i, v in enumerate(classes):
        res[int(v)] = int(class_weight[i] * configuration["sampling"]["favorisationCoeff"]) if v > 1 \
            else int(class_weight[i])
    sys.stdout.write(reports.tabs + 'Class weights : ' + str(res))
    return res
