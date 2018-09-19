import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, GRU, LSTM
from keras.models import Model
from keras.utils import to_categorical

import reports
from reports import *
from transitions import TransitionType


class Network:
    def __init__(self, normalizer):
        sys.stdout.write('Deep model(compositional) \n')
        embConf = configuration["model"]["embedding"]
        paddingConf = configuration["model"]["padding"]
        elemNum = paddingConf["s0Padding"] + paddingConf["s1Padding"] + paddingConf["bPadding"]
        concLayers, inputLayers = [], []
        if embConf["active"]:
            if embConf["concatenation"]:
                wordLayer, flattenLayer = elemModule(elemNum, normalizer)
                inputLayers.append(wordLayer)
                concLayers.append(flattenLayer)
            else:
                if embConf["usePos"]:
                    posLayer, posflattenLayer = elemModule(elemNum, normalizer, usePos=True)
                    inputLayers.append(posLayer)
                    concLayers.append(posflattenLayer)
                tokenLayer, tokenflattenLayer = elemModule(elemNum, normalizer, useToken=True)
                inputLayers.append(tokenLayer)
                concLayers.append(tokenflattenLayer)
        if configuration["features"]["active"]:
            featureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='Features')
            inputLayers.append(featureLayer)
            concLayers.append(featureLayer)
        # Merge layer
        concLayer = concLayers[0] if len(concLayers) == 1 else keras.layers.concatenate(concLayers)
        # MLP module
        output = mlpModule(concLayer)
        self.model = Model(inputs=inputLayers, outputs=output)
        sys.stdout.write('# Parameters = {0}\n'.format(self.model.count_params()))
        print self.model.summary()
        reports.saveNetwork(self.model)

    def predict(self, trans, normalizer):
        useEmbedding = configuration["model"]["embedding"]["active"]
        useConcatenation = configuration["model"]["embedding"]["concatenation"]
        usePos = configuration["model"]["embedding"]["usePos"]
        useFeatures = configuration["features"]["active"]

        dataEntry = normalizer.normalize(trans, useFeatures=useFeatures, useEmbedding=useEmbedding,
                                         useConcatenation=useConcatenation, usePos=usePos)
        inputVec = []
        for i in range(normalizer.inputListDimension):
            inputVec.append(np.asarray([dataEntry[i]]))
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=configuration["model"]["predict"]["verbose"])
        return oneHotRep[0]


def train(model, normalizer, corpus):
    trainConf = configuration["model"]["train"]
    model.compile(loss=trainConf["loss"], optimizer=trainConf["optimizer"], metrics=['accuracy'])
    bestWeightPath = reports.getBestWeightFilePath()
    callbacks = [
        ModelCheckpoint(bestWeightPath, monitor=trainConf["monitor"], verbose=1, save_best_only=True, mode='max')
    ] if bestWeightPath else []
    if trainConf["earlyStop"]:
        callbacks.append(EarlyStopping(monitor=trainConf["monitor"], min_delta=trainConf["minDelta"],
                                       patience=2, verbose=trainConf["verbose"]))
    labels, data = normalizer.generateLearningData(corpus)
    labels = to_categorical(labels, num_classes=len(TransitionType))
    history = model.fit(data, labels, validation_split=trainConf["validationSplit"],
                        epochs=trainConf["epochs"],
                        batch_size=trainConf["batchSize"],
                        verbose=trainConf["verbose"],
                        callbacks=callbacks)
    # TODO : Add sample weights to train
    validationData = []
    if normalizer.inputListDimension != 1:
        for dataTensor in data:
            validationData.append(dataTensor[int(len(dataTensor) * (1 - trainConf["validationSplit"])):])
    else:
        validationData = data[int(len(data) * (1 - trainConf["validationSplit"])):]
    validationLabel = labels[int(len(labels) * (1 - trainConf["validationSplit"])):]
    history = model.fit(validationData, validationLabel, epochs=len(history.epoch), batch_size=trainConf["batchSize"],
                        verbose=trainConf["verbose"])
    # reports.saveHistory(history)
    # plot(history)
    reports.saveModel(model)


def lstmModule(unitNum, name, sharedEmbedding, sharedLSTM, sharedLSTM2=None):
    inputLayer = Input(shape=(unitNum,), name=name + 'InputLayer')
    if sharedLSTM2:
        outputLayer = sharedLSTM(sharedLSTM2(sharedEmbedding(inputLayer)))
    else:
        outputLayer = sharedLSTM(sharedEmbedding(inputLayer))
    return inputLayer, outputLayer


def mlpModule(inputLayer):
    mlpConf = configuration["model"]["mlp"]
    lastLayer = inputLayer
    dense1Conf = mlpConf["dense1"]
    if dense1Conf["active"]:
        dense1Layer = Dense(dense1Conf["unitNumber"], activation=dense1Conf["activation"])(inputLayer)
        dropoutLayer = Dropout(0.2)(dense1Layer)
        lastLayer = dropoutLayer
        dense2Conf = mlpConf["dense2"]
        if dense2Conf["active"]:
            dense2Layer = Dense(dense2Conf["unitNumber"], activation=dense2Conf["activation"])(lastLayer)
            dropout2Layer = Dropout(0.2)(dense2Layer)
            lastLayer = dropout2Layer
            dense3Conf = mlpConf["dense3"]
            if dense3Conf["active"]:
                dense3Layer = Dense(dense3Conf["unitNumber"], activation=dense3Conf["activation"])(lastLayer)
                dropout3Layer = Dropout(0.2)(dense3Layer)
                lastLayer = dropout3Layer
    return Dense(len(TransitionType), activation='softmax', name='output')(lastLayer)


def elemModule(elemNum, normalizer, usePos=False, useToken=False):
    embConf = configuration["model"]["embedding"]
    name, inputDim, outputDim, weights = getInputConf(normalizer, usePos=usePos, useToken=useToken)
    wordLayer = Input(shape=(elemNum,), name=name)
    if embConf["initialisation"]["active"] and ((usePos and embConf["initialisation"]["pos"])
                                                or (useToken and embConf["initialisation"]["token"]) or not (
                    useToken or usePos)):
        sys.stdout.write('# {0} weight matrix = True\n'.format(name))
        embLayer = Embedding(inputDim, outputDim, weights=weights, trainable=True)(wordLayer)
    else:
        sys.stdout.write('# {0} weight matrix = False\n'.format(name))
        embLayer = Embedding(inputDim, outputDim)(wordLayer)
    rnnLayer = rnnModule(embLayer)
    if rnnLayer:
        return wordLayer, rnnLayer
    flattenLayer = Flatten(name='{0}Flatten'.format(name))(embLayer)
    return wordLayer, flattenLayer


def rnnModule(embLayer):
    rnnConfig = configuration["model"]["rnn"]
    if rnnConfig["active"]:
        if rnnConfig["gru"]:
            if rnnConfig["stacked"]:
                gru1 = GRU(rnnConfig["rnn1"]["unitNumber"], name='gru 1', return_sequences=True)(embLayer)
                gru2 = GRU(rnnConfig["rnn2"]["unitNumber"], name='gru 2')(gru1)
                return gru2
            else:
                gru1 = GRU(rnnConfig["rnn1"]["unitNumber"], name='gru 1')(embLayer)
                return gru1
        else:
            if rnnConfig["stacked"]:
                lstm1 = LSTM(rnnConfig["rnn1"]["unitNumber"], name='lstm 1', return_sequences=True)(embLayer)
                lstm2 = LSTM(rnnConfig["rnn2"]["unitNumber"], name='lstm 2')(lstm1)
                return lstm2
            else:
                lstm1 = LSTM(rnnConfig["rnn1"]["unitNumber"], name='lstm 1')(embLayer)
                return lstm1
    return None


def getInputConf(normalizer, usePos=False, useToken=False):
    embConf, weights = configuration["model"]["embedding"], []
    if usePos:
        name = 'pos'
        inputDim = len(normalizer.vocabulary.posIndices)
        outputDim = embConf["posEmb"]
        if embConf["initialisation"]["active"] and embConf["initialisation"]["pos"]:
            weights = [normalizer.posWeightMatrix]
    elif useToken:
        name = 'token'
        inputDim = len(normalizer.vocabulary.tokenIndices)
        outputDim = embConf["tokenEmb"]
        if embConf["initialisation"]["active"] and embConf["initialisation"]["token"]:
            weights = [normalizer.tokenWeightMatrix]
    # Buffer-based Embedding Module
    else:
        name = 'words'
        inputDim = len(normalizer.vocabulary.indices)
        outputDim = normalizer.vocabulary.embDim
        if embConf["initialisation"]["active"]:
            weights = [normalizer.weightMatrix]
    return name, inputDim, outputDim, weights


def plot(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    modelAccPath = os.path.join(reports.XP_CURRENT_DIR_PATH, 'accuracy-epoch.png')
    if not configuration["evaluation"]["cv"]["active"]:
        modelAccPath = os.path.join(reports.XP_CURRENT_DIR_PATH,
                                    str(not configuration["evaluation"]["cv"]["currentIter"]),
                                    'accuracy-epoch.png')
    plt.savefig(modelAccPath)
    # plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    modelLossPath = os.path.join(reports.XP_CURRENT_DIR_PATH, 'loss-epochs.png')
    if not configuration["evaluation"]["cv"]["active"]:
        modelLossPath = os.path.join(reports.XP_CURRENT_DIR_PATH,
                                     str(not configuration["evaluation"]["cv"]["currentIter"]),
                                     'loss-epochs.png')
    plt.savefig(modelLossPath)
    # plt.show()
