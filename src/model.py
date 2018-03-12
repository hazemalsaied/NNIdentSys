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
from vocabulary import unknown, number, Vocabulary

CLASS_NUM = len(TransitionType)
OPTIMIZER = 'rmsprop'
ADAM_OPTIMIZER = 'adam'
BEST_WEIGHT_FILE = 'bestWeigths.hdf5'
LOSS = 'categorical_crossentropy'


class Network(object):
    def __init__(self):
        reports.saveNetwork(self.model)

    def train(self, normalizer, corpus):
        print self.model.summary()
        if settings.USE_ADAM:
            self.model.compile(loss=LOSS, optimizer=ADAM_OPTIMIZER, metrics=['accuracy'])
        else:
            self.model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])
        bestWeightPath = os.path.join(reports.XP_CURRENT_DIR_PATH, BEST_WEIGHT_FILE)
        if settings.XP_CROSS_VALIDATION:
            bestWeightPath = os.path.join(reports.XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION),
                                          BEST_WEIGHT_FILE)
        callbacks = [
            ModelCheckpoint(bestWeightPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        ] if not settings.XP_LOAD_MODEL or not settings.XP_DEBUG_DATA_SET else []
        if settings.EARLY_STOP:
            callbacks.append(
                EarlyStopping(monitor='val_acc', min_delta=.5, patience=2, verbose=settings.NN_VERBOSE))
        time = datetime.datetime.now()
        logging.warn('Training started!')
        labels, data = normalizer.generateLearningData(corpus)
        labels = to_categorical(labels, num_classes=CLASS_NUM)
        history = self.model.fit(data, labels, validation_split=0.2, epochs=settings.NN_EPOCHS,
                                 batch_size=settings.NN_BATCH_SIZE,
                                 verbose=settings.NN_VERBOSE,
                                 callbacks=callbacks)
        logging.warn('Training has taken: {0}!'.format(datetime.datetime.now() - time))
        reports.saveHistory(history)
        if not settings.USE_CLUSTER:
            self.plotTraining(history)
        reports.saveModel(self.model)

    def plotTraining(self, history):
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        modelAccPath = os.path.join(reports.XP_CURRENT_DIR_PATH, 'accuracy-epoch.png')
        if settings.XP_CROSS_VALIDATION:
            modelAccPath = os.path.join(reports.XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION),
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
        if settings.XP_CROSS_VALIDATION:
            modelLossPath = os.path.join(reports.XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION),
                                         'loss-epochs.png')
        plt.savefig(modelLossPath)
        # plt.show()

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
        dense1Layer = Dense(settings.MLP_LAYER_1_UNIT_NUM, activation=Network.getActiviation1())(inputLayer)
        dropoutLayer = Dropout(0.2)(dense1Layer)
        if settings.USE_DENSE_2:
            dense2Layer = Dense(settings.MLP_LAYER_2_UNIT_NUM, activation=Network.getActiviation1())(dropoutLayer)
            dropout2Layer = Dropout(0.2)(dense2Layer)
        else:
            mainOutputLayer = Dense(CLASS_NUM, activation='softmax', name='mainOutputLayer')(dropoutLayer)
            return mainOutputLayer
        if settings.USE_DENSE_3:
            dense3Layer = Dense(settings.MLP_LAYER_2_UNIT_NUM, activation=Network.getActiviation2())(dropout2Layer)
            dropout3Layer = Dropout(0.2)(dense3Layer)
            mainOutputLayer = Dense(CLASS_NUM, activation='softmax', name='mainOutputLayer')(dropout3Layer)
        else:
            mainOutputLayer = Dense(CLASS_NUM, activation='softmax', name='mainOutputLayer')(dropout2Layer)
        return mainOutputLayer

    @staticmethod
    def createEmbeddingModule(wordNum, normalizer):
        # Buffer-based Embedding Module
        wordLayer = Input(shape=(wordNum,), name='word')
        if settings.INITIALIZE_EMBEDDING:
            embLayer = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size,
                                 weights=[normalizer.weightMatrix], trainable=settings.TRAINABLE_EMBEDDING)(wordLayer)
        else:
            embLayer = Embedding(output_dim=normalizer.vocabulary.embDim, input_dim=normalizer.vocabulary.size)(
                wordLayer)
        flattenLayer = Flatten(name='flatten')(embLayer)
        return wordLayer, flattenLayer

    @staticmethod
    def createPOSEmbeddingModule(wordNum, normalizer):
        # Buffer-based Embedding Module
        wordLayer = Input(shape=(wordNum,), name='word')
        if settings.INITIALIZE_EMBEDDING:
            embLayer = Embedding(output_dim=normalizer.vocabulary.postagDim, input_dim=len(normalizer.vocabulary.posIndices),
                                 weights=[normalizer.posWeightMatrix], trainable=settings.TRAINABLE_EMBEDDING)(wordLayer)
        else:
            embLayer = Embedding(output_dim=normalizer.vocabulary.postagDim, input_dim=len(normalizer.vocabulary.posIndices))(
                wordLayer)
        flattenLayer = Flatten(name='flatten')(embLayer)
        return wordLayer, flattenLayer

    @staticmethod
    def createTokenEmbeddingModule(wordNum, normalizer):
        # Buffer-based Embedding Module
        wordLayer = Input(shape=(wordNum,), name='word')
        if settings.INITIALIZE_EMBEDDING:
            embLayer = Embedding(output_dim=normalizer.vocabulary.tokenDim,
                                 input_dim=len(normalizer.vocabulary.tokenIndices),
                                 weights=[normalizer.tokenWeightMatrix], trainable=settings.TRAINABLE_EMBEDDING)(
                wordLayer)
        else:
            embLayer = Embedding(output_dim=normalizer.vocabulary.tokenDim,
                                 input_dim=len(normalizer.vocabulary.tokenIndices))(
                wordLayer)
        flattenLayer = Flatten(name='flatten')(embLayer)
        return wordLayer, flattenLayer

    @staticmethod
    def getActiviation1():
        if settings.MLP_USE_RELU_1:
            return 'relu'
        if settings.MLP_USE_TANH_1:
            return 'tanh'
        return 'relu'

    @staticmethod
    def getActiviation2():
        if settings.MLP_USE_RELU_2:
            return 'relu'
        if settings.MLP_USE_SIGMOID_2:
            return 'sigmoid'
        if settings.MLP_USE_TANH_2:
            return 'tanh'
        return 'relu'


class Normalizer(object):
    def __init__(self, corpus, INPUT_LIST_NUM, extractor=True):
        self.vocabulary = Vocabulary(corpus)

        if settings.USE_SEPERATED_EMB_MODULE:
            self.posWeightMatrix = np.zeros((len(self.vocabulary.posIndices), self.vocabulary.postagDim))
            for tokenKey in self.vocabulary.posIndices:
                self.posWeightMatrix[self.vocabulary.posIndices[tokenKey]] = self.vocabulary.posEmbeddings[tokenKey]
            del self.vocabulary.posEmbeddings

            self.tokenWeightMatrix = np.zeros((len(self.vocabulary.tokenIndices), self.vocabulary.tokenDim))
            for tokenKey in self.vocabulary.tokenIndices:
                self.tokenWeightMatrix[self.vocabulary.tokenIndices[tokenKey]] = self.vocabulary.tokenEmbeddings[tokenKey]
            del self.vocabulary.tokenEmbeddings

        else:
            self.weightMatrix = np.zeros((len(self.vocabulary.indices), self.vocabulary.embDim))
            for tokenKey in self.vocabulary.indices:
                self.weightMatrix[self.vocabulary.indices[tokenKey]] = self.vocabulary.embeddings[tokenKey]
            del self.vocabulary.embeddings
            self.inputListDimension = INPUT_LIST_NUM
        if extractor:
            self.nnExtractor = Extractor(corpus)
        reports.saveNormalizer(self)

    def generateLearningData(self, corpus):
        data, labels = [], []
        if self.inputListDimension != 1:
            for i in range(self.inputListDimension):
                data.append(list())
        for sent in corpus:
            trans = sent.initialTransition
            while trans.next:
                dataEntry = self.normalize(trans)
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

    def normalize(self, trans):
        pass

    def getIndices(self, tokens):
        result = []
        for token in tokens:
            key = self.getKey(token)
            result.append(self.vocabulary.indices[key])
        return np.asarray(result)

    def getKey(self, token):
        if any(ch.isdigit() for ch in token.getTokenOrLemma()):
            key = number
            if settings.USE_POS_EMB:
                posTag = unknown if token.posTag not in self.vocabulary.posIndices else token.posTag
                key += '_' + posTag
            return key
        key = token.getStandardKey()
        if key in self.vocabulary.indices:
            return key
        tokenTxt = unknown if token.getTokenOrLemma() not in self.vocabulary.tokenIndices else token.getTokenOrLemma()
        if settings.USE_POS_EMB:
            posTag = unknown if token.posTag not in self.vocabulary.posIndices else token.posTag

        if settings.USE_POS_EMB:
            if tokenTxt + '_' + posTag in self.vocabulary.indices:
                return tokenTxt + '_' + posTag
            if tokenTxt + '_' + unknown in self.vocabulary.indices:
                return tokenTxt + '_' + unknown
            if unknown + '_' + posTag in self.vocabulary.indices:
                return unknown + '_' + posTag
            return unknown + '_' + unknown
        else:
            if tokenTxt in self.vocabulary.indices:
                return tokenTxt
            return unknown
