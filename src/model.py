import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input
from keras.utils import to_categorical

import reports
from extraction import NNExtractor
from reports import *
from transitions import TransitionType
from vocabulary import unknown, number, Vocabulary

EPOCHS = 15
BATCH_SIZE = 128
TRAIN_VERBOSE = 2
PREDICT_VERBOSE = 0
EARLY_STOP = True
SAVE_MODEL = True
OPTIMIZER = 'rmsprop'


class Network(object):
    def __init__(self):
        reports.saveNetwork(self.model)

    def train(self, normalizer, corpus):
        print self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
        MODEL_WEIGHT_FILE = os.path.join(reports.XP_CURRENT_DIR_PATH, 'bestWeigths.hdf5')
        if settings.XP_CROSS_VALIDATION:
            MODEL_WEIGHT_FILE = os.path.join(reports.XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION),
                                             'bestWeigths.hdf5')
        callbacks = [ModelCheckpoint(MODEL_WEIGHT_FILE, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')] if SAVE_MODEL and not settings.XP_CROSS_VALIDATION else []
        if settings.EARLY_STOP:
            earlyStopping = EarlyStopping(monitor='val_acc', min_delta=.5, patience=2, verbose=TRAIN_VERBOSE)
            callbacks.append(earlyStopping)
        time = datetime.datetime.now()
        labels, data = normalizer.generateLearningData(corpus)
        labels = to_categorical(labels, num_classes=len(TransitionType))
        logging.warn('Deep Model training started!')
        history = self.model.fit(data, labels, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                 verbose=TRAIN_VERBOSE,
                                 callbacks=callbacks)
        # historyFile = os.path.join(XP_CURRENT_DIR_PATH, 'history.txt')
        # if settings.XP_CROSS_VALIDATION:
        #     historyFile = os.path.join(XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION), 'history.txt')
        # with open(historyFile, 'w') as f:
        #     str = str(history.history['acc']) + str(history.history['val_acc']) + str(plt.ylabel('accuracy')) + str(plt.xlabel('epoch'))
        #     f.write(str)
        # # summarize history for accuracy
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # modelAccPath = os.path.join(reports.XP_CURRENT_DIR_PATH, 'accuracy-epoch.png')
        # if settings.XP_CROSS_VALIDATION:
        #     modelAccPath = os.path.join(reports.XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION),
        #                                      'accuracy-epoch.png')
        # plt.savefig(modelAccPath)
        # #plt.show()
        # # summarize history for loss
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # modelLossPath = os.path.join(reports.XP_CURRENT_DIR_PATH, 'loss-epochs.png')
        # if settings.XP_CROSS_VALIDATION:
        #     modelLossPath = os.path.join(reports.XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION),
        #                                 'loss-epochs.png')
        # plt.savefig(modelLossPath)
        # #plt.show()

        logging.warn('Training has taken: {0}!'.format(datetime.datetime.now() - time))
        reports.saveModel(self.model)

    def predict(self, trans, normalizer):
        pass

    def createLSTMModule(self, unitNum, name, sharedEmbedding, sharedLSTM, sharedLSTM2=None):
        inputLayer = Input(shape=(unitNum,), name=name + 'InputLayer')
        if sharedLSTM2:
            outputLayer = sharedLSTM(sharedLSTM2(sharedEmbedding(inputLayer)))
        else:
            outputLayer = sharedLSTM(sharedEmbedding(inputLayer))
        return inputLayer, outputLayer

    def getActiviation1(self):
        if settings.MLP_USE_RELU_1:
            return 'relu'
        if settings.MLP_USE_TANH_1:
            return 'tanh'
        return 'relu'

    def getActiviation2(self):
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
        self.weightMatrix = np.zeros((len(self.vocabulary.indices), self.vocabulary.embDim))
        for tokenKey in self.vocabulary.indices:
            self.weightMatrix[self.vocabulary.indices[tokenKey]] = self.vocabulary.embeddings[tokenKey]
        del self.vocabulary.embeddings
        self.inputListDimension = INPUT_LIST_NUM
        if extractor:
            self.nnExtractor = NNExtractor(corpus)
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
            posTag = unknown if token.posTag not in self.vocabulary.posIndices else token.posTag
            return number + '_' + posTag
        key = token.getStandardKey()
        if key in self.vocabulary.indices:
            return key
        tokenTxt = unknown if token.getTokenOrLemma() not in self.vocabulary.tokenIndices else token.getTokenOrLemma()
        posTag = unknown if token.posTag not in self.vocabulary.posIndices else token.posTag
        if tokenTxt + '_' + posTag in self.vocabulary.indices:
            return tokenTxt + '_' + posTag
        if tokenTxt + '_' + unknown in self.vocabulary.indices:
            return tokenTxt + '_' + unknown
        if unknown + '_' + posTag in self.vocabulary.indices:
            return unknown + '_' + posTag
        return unknown + '_' + unknown
