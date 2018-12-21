from collections import Counter
from random import uniform

import keras
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import reports
import sampling
from corpus import getRelevantModelAndNormalizer
from corpus import getTokens
from extraction import Extractor
from modelLinear import getFeatures
from reports import *
from transitions import TransitionType
from wordEmbLoader import empty
from wordEmbLoader import unk, number

enableCategorization = False

configuration['convo'] = {
    'inputItems': 10,
    'tokenEmb': 100,
    'posEmb': 25,
    'trainable': True,
    'conv2D': 32,
    's1TokenNum': 4,
    's0TokenNum': 4,
    'bTokenNum': 1,
    'denseUnits': 128,
    'denseAct': 'relu',
    'denseDrop': .2,


    }
convoConf = configuration['convo']


class Network:
    def __init__(self, corpus, linearInMLP=False):
        self.vocabulary = Vocabulary(corpus)
        self.nnExtractor = Extractor(corpus) if configuration['convo']['features'] else None
        input, output = self.createTheModel(linearInMLP=linearInMLP)
        self.model = Model(inputs=input, outputs=output)
        if configuration['others']['verbose']:
            sys.stdout.write('# Parameters = {0}\n'.format(self.model.count_params()))
            print self.model.summary()

    def createTheModel(self, linearInMLP=False):
        inputLayers, concLayers = [], []
        inputToken = Input((configuration['convo']['inputItems'],))
        inputLayers.append(inputToken)
        tokenEmb = Embedding(len(self.vocabulary.tokenIndices), configuration['convo']['tokenEmb'], trainable=configuration['convo']['trainable'])(inputToken)
        tokenFlatten = Flatten()(tokenEmb)
        concLayers.append(tokenFlatten)
        inputPos = Input((configuration['convo']['inputItems'],))
        inputLayers.append(inputPos)
        posEmb = Embedding(len(self.vocabulary.posIndices), configuration['convo']['posEmb'],trainable=configuration['convo']['trainable'])(inputPos)
        posFlatten = Flatten()(posEmb)
        concLayers.append(posFlatten)

        conc = keras.layers.concatenate(concLayers) if len(concLayers) > 1 else concLayers[0]
        convo = Conv2D(configuration['convo']['conv2D'], kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape)(conc)
        pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convo)
        dense = Dense(configuration['convo']['denseUnits'],
                      activation=configuration['cono']['denseAct'],
                      dropout=configuration['cono']['denseDrop'])(pool)
        softmaxLayer = Dense(8 if enableCategorization else 4, activation='softmax')(dense)
        return inputLayers, softmaxLayer

    def predict(self, trans, linearModels=None, linearVecs=None):
        inputs = []
        tokenIdxs, posIdxs = self.getAttachedIndices(trans)
        inputs.append(np.asarray([tokenIdxs]))
        inputs.append(np.asarray([posIdxs]))
        if linearModels and linearVecs:
            linearModel, linearVec = getRelevantModelAndNormalizer(trans.sent, None, linearModels, linearVecs, True)
            featDic = getFeatures(trans, trans.sent)
            predTrans = linearModel.predict(linearVec.transform(featDic))[0]
            predVec = [1 if t.value == predTrans else 0 for t in TransitionType]
            inputs.append(np.asarray([predVec]))
        if configuration['convo']['features']:
            features = np.asarray(self.nnExtractor.vectorize(trans))
            inputs.append(np.asarray([features]))
        oneHotRep = self.model.predict(inputs, batch_size=1,
                                       verbose=configuration['convo']['predictVerbose'])
        return oneHotRep[0]

    def getLearningData(self, corpus):
        labels, data = [], []
        for sent in corpus.trainingSents:
            trans = sent.initialTransition
            while trans and trans.next:
                if not configuration['sampling']['importantTransitions'] or trans.isImportant():
                    tokenIdxs, posIdxs = self.getIdxs(trans)
                    data.append(np.asarray(np.concatenate((tokenIdxs, posIdxs))))
                    labels.append(trans.next.type.value if trans.next.type.value <= 2 else (
                        trans.next.type.value if enableCategorization else 3))
                trans = trans.next
        return labels, data

    def getIdxs(self, config):
        idxs, posIdxs = [], []
        tokenVocab, posVocab = self.vocabulary.tokenIndices, self.vocabulary.posIndices
        if config.stack and len(config.stack) > 1:
            for t in getTokens(config.stack[-2])[:convoConf['s1TokenNum']]:
                if t.isNumber():
                    idxs.append(tokenVocab[number])
                else:
                    idxs.append(
                        tokenVocab[t.getTokenOrLemma()] if t.getTokenOrLemma() in tokenVocab else tokenVocab[unk])
                posIdxs.append(posVocab[t.posTag.lower()] if t.posTag.lower() in posVocab else posVocab[unk])
        while len(idxs) < convoConf['s1TokenNum']:
            idxs = idxs + [tokenVocab[empty]]
            posIdxs = posIdxs + [posVocab[empty]]

        if config.stack:

            for t in getTokens(config.stack[-1])[:convoConf['s0TokenNum']]:
                if t.isNumber():
                    idxs.append(tokenVocab[number])
                else:
                    idxs.append(
                        tokenVocab[t.getTokenOrLemma()] if t.getTokenOrLemma() in tokenVocab else tokenVocab[unk])
                posIdxs.append(posVocab[t.posTag.lower()] if t.posTag.lower() in posVocab else posVocab[unk])
        while len(idxs) < convoConf['s0TokenNum'] + convoConf['s1TokenNum']:
            idxs = idxs + [tokenVocab[empty]]
            posIdxs = posIdxs + [posVocab[empty]]

        if config.buffer:
            for t in config.buffer[:convoConf['bTokenNum']]:
                if t.isNumber():
                    idxs.append(tokenVocab[number])
                else:
                    idxs.append(
                        tokenVocab[t.getTokenOrLemma()] if t.getTokenOrLemma() in tokenVocab else tokenVocab[unk])
                posIdxs.append(posVocab[t.posTag.lower()] if t.posTag.lower() in posVocab else posVocab[unk])
        while len(idxs) < convoConf['s0TokenNum'] + convoConf['s1TokenNum'] + convoConf['bTokenNum']:
            idxs = idxs + [tokenVocab[empty]]
            posIdxs = posIdxs + [posVocab[empty]]
        return np.asarray(idxs), np.asarray(posIdxs)


    def getAttachedIndices(self, trans):
        emptyTokenIdx = self.vocabulary.tokenIndices[empty]
        emptyPosIdx = self.vocabulary.posIndices[empty]
        tokenIdxs, posIdxs = [], []
        if trans.configuration.stack:
            s0Tokens = getTokens(trans.configuration.stack[-1])
            tokenIdx, posIdx = self.vocabulary.getIndices(s0Tokens)
            tokenIdxs.append(tokenIdx)
            posIdxs.append(posIdx)
            if len(trans.configuration.stack) > 1:
                s1Tokens = getTokens(trans.configuration.stack[-2])
                tokenIdx, posIdx = self.vocabulary.getIndices(s1Tokens)
                tokenIdxs.append(tokenIdx)
                posIdxs.append(posIdx)
            else:
                tokenIdxs.append(emptyTokenIdx)
                posIdxs.append(emptyPosIdx)
        else:
            tokenIdxs.append(emptyTokenIdx)
            tokenIdxs.append(emptyTokenIdx)
            posIdxs.append(emptyPosIdx)
            posIdxs.append(emptyPosIdx)

        if trans.configuration.buffer:
            tokenIdx, posIdx = self.vocabulary.getIndices([trans.configuration.buffer[0]])
            tokenIdxs.append(tokenIdx)
            posIdxs.append(posIdx)
            if configuration['convo']['inputItems'] == 4:
                if len(trans.configuration.buffer) > 1:
                    tokenIdx, posIdx = self.vocabulary.getIndices([trans.configuration.buffer[1]])
                    tokenIdxs.append(tokenIdx)
                    posIdxs.append(posIdx)
                else:
                    tokenIdxs.append(emptyTokenIdx)
                    posIdxs.append(emptyPosIdx)
        else:
            tokenIdxs.append(emptyTokenIdx)
            posIdxs.append(emptyPosIdx)
            if configuration['convo']['inputItems'] == 4:
                tokenIdxs.append(emptyTokenIdx)
                posIdxs.append(emptyPosIdx)

        return np.asarray(tokenIdxs), np.asarray(posIdxs)

    def train(self, corpus, linearModels=None, linearNormalizers=None):
        labels, data = self.getLearningData(corpus)
        if configuration['others']['verbose']:
            sys.stdout.write(reports.seperator + reports.tabs + 'Sampling' + reports.doubleSep)
        if configuration['sampling']['focused']:
            data, labels = sampling.overSampleImporTrans(data, labels, corpus, self.vocabulary)
        labels, data = sampling.overSample(labels, data, linearInMlp=True)
        if configuration['convo']['earlyStop']:
            # To make sure that we will get a random validation dataset
            labelsAndData = sampling.shuffleArrayInParallel(
                [labels, data[0], data[1], data[2]] if linearModels else [labels, data[0], data[1]])
            labels = labelsAndData[0]
            data = labelsAndData[1:]
        if configuration['others']['verbose']:
            lblDistribution = Counter(labels)
            sys.stdout.write(tabs + '{0} Labels in train : {1}\n'.format(len(lblDistribution), lblDistribution))
        if configuration['others']['verbose']:
            valDistribution = Counter(labels[int(len(labels) * (1 - configuration['convo']['validationSplit'])):])
            sys.stdout.write(tabs + '{0} Labels in valid : {1}\n'.format(len(valDistribution), valDistribution))
        self.classWeightDic = sampling.getClassWeights(labels)
        sampleWeights = sampling.getSampleWeightArray(labels, self.classWeightDic)
        labels = to_categorical(labels, num_classes=8 if enableCategorization else 4)
        self.model.compile(loss=configuration['convo']['loss'], optimizer=getOptimizer(), metrics=['accuracy'])
        history = self.model.fit(data, labels,
                                 validation_split=configuration['convo']['validationSplit'],
                                 epochs=configuration['convo']['epochs'],
                                 batch_size=configuration['convo']['batchSize'],
                                 verbose=2 if configuration['others']['verbose'] else 0,
                                 callbacks=getCallBacks(),
                                 sample_weight=sampleWeights)
        if configuration['convo']['verbose']:
            sys.stdout.write('Epoch Losses= ' + str(history.history['loss']))
        self.trainValidationData(data, labels, history)

    def trainValidationData(self, data, labels, history):
        data, labels = getValidationData(data, labels)
        validationLabelsAsInt = [np.where(r == 1)[0][0] for r in labels]
        sampleWeights = sampling.getSampleWeightArray(validationLabelsAsInt, self.classWeightDic)
        history = self.model.fit(data, labels,
                                 epochs=len(history.epoch),
                                 batch_size=configuration['convo']['batchSize'],
                                 verbose=0,
                                 sample_weight=sampleWeights)
        return history


def getValidationData(data, labels):
    trainConf = configuration['convo']
    validationData = []
    for dataTensor in data:
        validationData.append(dataTensor[int(len(dataTensor) * (1 - trainConf['validationSplit'])):])
    validationLabel = labels[int(len(labels) * (1 - trainConf['validationSplit'])):]
    return validationData, validationLabel


def getOptimizer():
    if configuration['others']['verbose']:
        sys.stdout.write(reports.seperator + reports.tabs +
                         'Optimizer : Adagrad,  learning rate = {0}'.format(configuration['convo']['lr'])
                         + reports.seperator)
    return optimizers.Adagrad(lr=configuration['convo']['lr'], epsilon=None, decay=0.0)


def getCallBacks():
    trainConf = configuration['convo']
    bestWeightPath = reports.getBestWeightFilePath()
    callbacks = [
        ModelCheckpoint(bestWeightPath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    ] if bestWeightPath else []
    if trainConf['earlyStop']:
        es = EarlyStopping(monitor='val_loss',
                           min_delta=trainConf['minDelta'],
                           patience=2,
                           verbose=trainConf['verbose'])
        callbacks.append(es)
    return callbacks


class Vocabulary:
    def __init__(self, corpus):
        self.tokenFreqs, self.posFreqs = getFrequencyDics(corpus)
        self.tokenIndices = indexateDic(self.tokenFreqs)
        self.posIndices = indexateDic(self.posFreqs)
        if configuration['convo']['verbose'] == 1:
            sys.stdout.write(str(self))
            self.verify(corpus)

    def __str__(self):
        res = seperator + tabs + 'Vocabulary' + doubleSep
        res += tabs + 'Tokens := {0} * POS : {1}'.format(len(self.tokenIndices), len(self.posIndices)) \
            if not configuration['xp']['compo'] else ''
        res += seperator
        return res

    def getEmptyIdx(self, getPos=False, getToken=False):
        pass

    def verify(self, corpus):
        importTokens = 0
        for t in corpus.mweTokenDictionary:
            if t not in self.tokenIndices:
                importTokens += 1
        if importTokens:
            sys.stdout.write(tabs + 'Important words not in vocabulary {0}\n'.format(importTokens))
        importMWEs = 0
        for mwe in corpus.mweDictionary:
            mwe = mwe.replace(' ', '_')
            if mwe not in self.tokenIndices:
                importMWEs += 1
                print mwe
        if importMWEs:
            sys.stdout.write(tabs + 'MWE not in vocabulary {0}\n'.format(importMWEs))
        if unk not in self.tokenIndices or empty not in self.tokenIndices:
            sys.stdout.write(tabs + 'unk or empty is not in vocabulary\n')
        dashedKeys = 0
        for k in self.tokenIndices:
            if '_' in k:
                dashedKeys += 1
        # supposedDashedKeys = 0
        # for mwe in corpus.mweDictionary:
        #     supposedDashedKeys += len(mwe.split(' ')) - 1
        # sys.stdout.write(tabs + 'Suupposed dashed keys in vocabulary {0}\n'.format(supposedDashedKeys))
        sys.stdout.write(tabs + 'Dashed keys in vocabulary {0}\n'.format(dashedKeys))
        oneFreq = 0
        for k in self.tokenFreqs:
            if self.tokenFreqs[k] == 1:
                oneFreq += 1
        sys.stdout.write(tabs + 'One occurrence keys in vocabulary {0} / {1}\n'.
                         format(dashedKeys, len(self.tokenFreqs)))


def getFrequencyDics(corpus, freqTaux=1):
    tokenVocab, posVocab = {unk: freqTaux + 1, empty: freqTaux + 1}, {unk: freqTaux + 1, empty: freqTaux + 1}
    for sent in corpus.trainingSents:
        trans = sent.initialTransition
        while trans:
            if trans.configuration.stack:
                tokens = getTokens(trans.configuration.stack[-1])
                if tokens:
                    tokenTxt, posTxt = attachTokens(tokens)
                    for c in tokenTxt:
                        if c.isdigit():
                            tokenTxt = number
                    tokenVocab[tokenTxt] = 1 if tokenTxt not in tokenVocab else tokenVocab[tokenTxt] + 1
                    posVocab[posTxt] = 1 if posTxt not in posVocab else posVocab[posTxt] + 1
            trans = trans.next
    if configuration['convo']['compactVocab']:
        sys.stdout.write(tabs + 'Compact Vocabulary cleaning:' + doubleSep)
        sys.stdout.write(tabs + 'Before : {0}\n'.format(len(tokenVocab)))
        for k in tokenVocab.keys():
            if k not in [empty, unk, number] and k.lower() not in corpus.mweTokenDictionary and '_' not in k:
                del tokenVocab[k]
        sys.stdout.write(tabs + 'After : {0}\n'.format(len(tokenVocab)))

    else:
        if configuration['others']['verbose']:
            sys.stdout.write(tabs + 'Non frequent word cleaning:' + doubleSep)
            sys.stdout.write(tabs + 'Before : {0}\n'.format(len(tokenVocab)))
        for k in tokenVocab.keys():
            if tokenVocab[k] <= freqTaux and '_' not in k and k.lower() not in corpus.mweTokenDictionary:
                if uniform(0, 1) < configuration['constants']['alpha']:
                    del tokenVocab[k]
        if configuration['others']['verbose']:
            sys.stdout.write(tabs + 'After : {0}\n'.format(len(tokenVocab)))
    return tokenVocab, posVocab


def attachTokens(tokens):
    tokenTxt, posTxt = '', ''
    for t in tokens:
        tokenTxt += t.getTokenOrLemma() + '_'
        posTxt += t.posTag + '_'
    return tokenTxt[:-1], posTxt[:-1].lower()


def indexateDic(dic):
    res = dict()
    r = range(len(dic))
    for i, k in enumerate(dic):
        res[k] = r[i]
    return res


def padSequence(seq, label, emptyIdx):
    padConf = configuration['convo']
    return np.asarray(pad_sequences([seq], maxlen=padConf[label], value=emptyIdx))[0]
