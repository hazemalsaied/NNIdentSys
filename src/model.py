import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Embedding
from keras.utils import to_categorical
from matplotlib import pyplot as plt

import reports
from extraction import Extractor
from reports import *
from transitions import TransitionType
from vocabulary import unk, number, Vocabulary

CLASS_NUM = len(TransitionType)


class AbstractNetwork(object):
    def __init__(self):
        reports.saveNetwork(self.model)

    def train(self, normalizer, corpus):
        trainConf = configuration["model"]["train"]
        print self.model.summary()
        self.model.compile(loss=trainConf["loss"], optimizer=trainConf["optimizer"], metrics=['accuracy'])
        bestWeightPath = reports.getBestWeightFilePath()
        callbacks = [
            ModelCheckpoint(bestWeightPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        ] if bestWeightPath else []
        if trainConf["earlyStop"]:
            callbacks.append(EarlyStopping(monitor='val_acc', min_delta=.5, patience=2, verbose=trainConf["verbose"]))
        time = datetime.datetime.now()
        logging.warn('Training started!')
        labels, data = normalizer.generateLearningData(corpus)
        labels = to_categorical(labels, num_classes=CLASS_NUM)
        history = self.model.fit(data, labels, validation_split=0.2,
                                 epochs=trainConf["epochs"],
                                 batch_size=trainConf["batchSize"],
                                 verbose=trainConf["verbose"],
                                 callbacks=callbacks)
        logging.warn('Training has taken: {0}!'.format(datetime.datetime.now() - time))
        # reports.saveHistory(history)
        if not configuration["evaluation"]["cluster"]:
            plotTraining(history)
        reports.saveModel(self.model)

    def predict(self, trans, normalizer):
        pass

    @staticmethod
    def createLSTMModule(unitNum, name, sharedEmbedding, sharedLSTM, sharedLSTM2=None):
        inputLayer = Input(shape=(unitNum,), name=name + 'InputLayer')
        if sharedLSTM2:
            outputLayer = sharedLSTM(sharedLSTM2(sharedEmbedding(inputLayer)))
        else:
            outputLayer = sharedLSTM(sharedEmbedding(inputLayer))
        return inputLayer, outputLayer

    @staticmethod
    def createMLPModule(inputLayer):
        mlpConf = configuration["model"]["topology"]["mlp"]
        dense1Layer = Dense(mlpConf["dense1"]["unitNumber"], activation=mlpConf["dense1"]["activation"])(inputLayer)
        dropoutLayer = Dropout(0.2)(dense1Layer)
        if mlpConf["dense2"]["active"]:
            dense2Layer = Dense(mlpConf["dense2"]["unitNumber"], activation=mlpConf["dense2"]["activation"])(
                dropoutLayer)
            dropout2Layer = Dropout(0.2)(dense2Layer)
        else:
            mainOutputLayer = Dense(CLASS_NUM, activation='softmax', name='mainOutputLayer')(dropoutLayer)
            return mainOutputLayer
        if mlpConf["dense3"]["active"]:
            dense3Layer = Dense(mlpConf["dense3"]["unitNumber"], activation=mlpConf["dense3"]["activation"])(
                dropout2Layer)
            dropout3Layer = Dropout(0.2)(dense3Layer)
            mainOutputLayer = Dense(CLASS_NUM, activation='softmax', name='mainOutputLayer')(dropout3Layer)
        else:
            mainOutputLayer = Dense(CLASS_NUM, activation='softmax', name='mainOutputLayer')(dropout2Layer)
        return mainOutputLayer

    @staticmethod
    def createEmbeddingModule(wordNum, normalizer):
        embConf = configuration["model"]["embedding"]
        # Buffer-based Embedding Module
        wordLayer = Input(shape=(wordNum,), name='word')
        if embConf["initialisation"]:
            embLayer = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                 weights=[normalizer.weightMatrix], trainable=True)(wordLayer)
        else:
            embLayer = Embedding(output_dim=embConf["posEmb"] + embConf["tokenEmb"],
                                 input_dim=normalizer.vocabulary.size)(
                wordLayer)
        flattenLayer = Flatten(name='flatten')(embLayer)
        return wordLayer, flattenLayer

    @staticmethod
    def createPOSEmbeddingModule(wordNum, normalizer):
        embConf = configuration["model"]["embedding"]
        # Buffer-based Embedding Module
        wordLayer = Input(shape=(wordNum,), name='pos')
        if embConf["initialisation"]:
            embLayer = Embedding(output_dim=normalizer.vocabulary.postagDim,
                                 input_dim=len(normalizer.vocabulary.posIndices),
                                 weights=[normalizer.posWeightMatrix], trainable=True)(wordLayer)
        else:
            embLayer = Embedding(output_dim=embConf["posEmb"], input_dim=len(normalizer.vocabulary.posIndices))(
                wordLayer)
        flattenLayer = Flatten(name='posFlatten')(embLayer)
        return wordLayer, flattenLayer

    @staticmethod
    def createTokenEmbeddingModule(wordNum, normalizer):
        embConf = configuration["model"]["embedding"]
        # Buffer-based Embedding Module
        wordLayer = Input(shape=(wordNum,), name='token')
        if embConf["initialisation"]:
            embLayer = Embedding(output_dim=normalizer.vocabulary.tokenDim,
                                 input_dim=len(normalizer.vocabulary.tokenIndices),
                                 weights=[normalizer.tokenWeightMatrix], trainable=True)(
                wordLayer)
        else:
            embLayer = Embedding(output_dim=embConf["tokenEmb"], input_dim=len(normalizer.vocabulary.tokenIndices)
                                 )(wordLayer)
        flattenLayer = Flatten(name='tokenFlatten')(embLayer)
        return wordLayer, flattenLayer


class AbstractNormalizer(object):
    def __init__(self, corpus,  extractor=True):
        self.vocabulary = Vocabulary(corpus)

        if not configuration["model"]["embedding"]["concatenation"]:
            self.posWeightMatrix = np.zeros((len(self.vocabulary.posIndices), self.vocabulary.postagDim))
            if not configuration["model"]["embedding"]["initialisation"]:
                for tokenKey in self.vocabulary.posIndices:
                    self.posWeightMatrix[self.vocabulary.posIndices[tokenKey]] = self.vocabulary.posEmbeddings[tokenKey]
            del self.vocabulary.posEmbeddings

            self.tokenWeightMatrix = np.zeros((len(self.vocabulary.tokenIndices), self.vocabulary.tokenDim))
            if not configuration["model"]["embedding"]["initialisation"]:
                for tokenKey in self.vocabulary.tokenIndices:
                    self.tokenWeightMatrix[self.vocabulary.tokenIndices[tokenKey]] = self.vocabulary.tokenEmbeddings[
                        tokenKey]
            del self.vocabulary.tokenEmbeddings

        else:
            self.weightMatrix = np.zeros((len(self.vocabulary.indices), self.vocabulary.embDim))
            if not configuration["model"]["embedding"]["initialisation"]:
                for tokenKey in self.vocabulary.indices:
                    self.weightMatrix[self.vocabulary.indices[tokenKey]] = self.vocabulary.embeddings[tokenKey]
            del self.vocabulary.embeddings
        #self.inputListDimension = INPUT_LIST_NUM
        if extractor:
            self.nnExtractor = Extractor(corpus)
        reports.saveNormalizer(self)

    def generateLearningData(self, corpus):
        data, labels = [], []
        useEmbedding = configuration["model"]["embedding"]["active"]
        useConcatenation = configuration["model"]["embedding"]["concatenation"]
        usePos = configuration["model"]["embedding"]["pos"]
        useFeatures = configuration["features"]["active"]
        logging.warn('Learning data generation has started with: Features: {0}; Embedding: {1}; concatenation: {2}; POS: {3}'.
            format(useFeatures,useConcatenation,useEmbedding, usePos))
        #if self.inputListDimension != 1:
        #    for i in range(self.inputListDimension):
        #       data.append(list())
        data = None
        for sent in corpus:
            trans = sent.initialTransition
            while trans.next:
                dataEntry = self.normalize(trans, useConcatenation=useConcatenation, useEmbedding=useEmbedding,
                                           usePos=usePos, useFeatures=useFeatures)
                # First iteration only
                if not data:
                    data = []
                    self.inputListDimension = len(dataEntry)
                    if len(dataEntry) >1:
                        for i in range(len(dataEntry)):
                            data.append(list())
                for i in range(self.inputListDimension):
                    if self.inputListDimension != 1:
                        data[i].append(dataEntry[i])
                    else:
                        data.append(dataEntry[i])
                labels = np.append(labels, trans.next.type.value)
                trans = trans.next
        if self.inputListDimension != 1:
            for i in range(len(data)):
                data[i] = np.asarray(data[i])
        if self.inputListDimension == 1:
            data = np.asarray(data)
        return np.asarray(labels), data

    def normalize(self, trans, useConcatenation=False, useEmbedding=False,
                  usePos=False, useFeatures=False):
        return []
    def getIndices(self, tokens, getPos=False, getToken=False):
        result = []
        for token in tokens:
            key = self.getKey(token, getPos=getPos, getToken=getToken)
            if getPos:
                result.append(self.vocabulary.posIndices[key])
            elif getToken:
                result.append(self.vocabulary.tokenIndices[key])
            else:
                result.append(self.vocabulary.indices[key])
        return np.asarray(result)

    def getKey(self, token, getPos=False, getToken=False):

        if getToken:
            if any(ch.isdigit() for ch in token.getTokenOrLemma()):
                key = number
                return key
            key = token.getStandardKey(getPos=False, getToken=True)
            if key in self.vocabulary.tokenIndices:
                return key
            return unk
        elif getPos:
            key = token.getStandardKey(getPos=True, getToken=False)
            if key in self.vocabulary.posIndices:
                return key
            return unk

        else:
            if any(ch.isdigit() for ch in token.getTokenOrLemma()):
                key = number
                posKey = token.getStandardKey(getPos=True, getToken=False)
                if key + '_' + posKey in self.vocabulary.indices:
                    return key + '_' + posKey
                return key + '_' + unk
            key = token.getStandardKey(getPos=False, getToken=False)
            if key in self.vocabulary.indices:
                return key
            return unk + '_' + unk


def plotTraining(history):
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
