from collections import Counter

import numpy as np
from keras.preprocessing.sequence import pad_sequences

import compoVocabulary
import nonCompoVocabulary
import reports
from corpus import getTokens
from extraction import Extractor
from reports import *
from wordEmbLoader import empty


class Normalizer:
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        self.inputListDimension = 2
        global embConf, initConf
        embConf = configuration["model"]["embedding"]
        initConf = embConf["initialisation"]
        self.vocabulary, self.posWeightMatrix, self.weightMatrix, self.tokenWeightMatrix = \
            None, None, None, None
        if embConf["active"]:
            if configuration["xp"]["compo"]:
                self.vocabulary = compoVocabulary.Vocabulary(corpus)
            else:
                self.vocabulary = nonCompoVocabulary.Vocabulary(corpus)
            self.allImportantLemmasInVocab(corpus)
            if initConf["active"]:
                if embConf["concatenation"]:
                    self.weightMatrix = initMatrix(self.vocabulary)
                    del self.vocabulary.embeddings
                else:
                    if embConf["usePos"]:
                        self.posWeightMatrix = initMatrix(self.vocabulary, usePos=True)
                        del self.vocabulary.posEmbeddings
                    self.tokenWeightMatrix = initMatrix(self.vocabulary, useToken=True)
                    del self.vocabulary.tokenEmbeddings
        self.nnExtractor = Extractor(corpus) if configuration["features"]["active"] else None
        reports.saveNormalizer(self)
        if configuration["xp"]["verbose"] == 1:
            sys.stdout.write(str(self))

    def allImportantLemmasInVocab(self, corpus):
        importantsAsUnk = 0
        for t in corpus.mweTokenDictionary:
            if t not in self.vocabulary.tokenIndices:
                importantsAsUnk += 1
        if importantsAsUnk:
            sys.stdout.write(tabs + 'Attention: important lemmas are not in vocbulary')
        return importantsAsUnk

    def __str__(self):
        report = seperator + tabs + 'Padding is activated' + doubleSep \
            if configuration["model"]["padding"]["active"] else ''
        if embConf["active"]:
            report += tabs + 'Embedding' + doubleSep
            report += tabs + 'Initialisation = {0}\n\tConcatenation = {1}\n'. \
                format(embConf["initialisation"]["type"] if embConf["initialisation"]["active"] else 'None',
                       embConf["concatenation"])
            report += tabs + '{0} : {1}\n\tPOS = {2} \n'. \
                format('Lemma' if embConf["lemma"] else 'Token',
                       embConf["tokenEmb"],
                       embConf["posEmb"] if embConf["usePos"] else 'None')
        return report

    def generateLearningData(self, corpus):
        global embConf
        embConf = configuration["model"]["embedding"]
        data, labels = [], []
        useEmbedding = embConf["active"]
        useConcatenation = embConf["concatenation"]
        usePos = embConf["usePos"]
        useFeatures = configuration["features"]["active"]
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
                    if len(dataEntry) > 1:
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

    def generateLearningDataAttached(self, corpus):
        labels, data = [], []
        for sent in corpus.trainingSents:
            trans = sent.initialTransition
            while trans and trans.next:
                if not configuration["sampling"]["importantTransitions"] or trans.isImportant():
                    tokenIdxs, posIdxs = self.getAttachedIndices(trans)
                    data.append(np.asarray(np.concatenate((tokenIdxs, posIdxs))))
                    labels.append(trans.next.type.value)
                trans = trans.next
        return labels, data

    def normalize(self, trans, useEmbedding=False, useConcatenation=False, usePos=False, useFeatures=False):
        results = []
        if useEmbedding:
            if useConcatenation:
                words = self.getIndices(trans)
                results.append(words)
            else:
                if usePos:
                    pos = self.getIndices(trans, getPos=True)
                    results.append(pos)
                tokens = self.getIndices(trans, getToken=True)
                results.append(tokens)
        if useFeatures:
            features = np.asarray(self.nnExtractor.vectorize(trans))
            results.append(features)
        return results

    def getIndices(self, trans, getPos=False, getToken=False):
        s0elems, s1elems, belems = [], [], []
        emptyIdx = self.vocabulary.getEmptyIdx(getPos=getPos, getToken=getToken)
        if trans.configuration.stack:
            s0Tokens = getTokens(trans.configuration.stack[-1])
            s0elems = self.vocabulary.getIndices(s0Tokens, getPos=getPos, getToken=getToken)
            if len(trans.configuration.stack) > 1:
                s1Tokens = getTokens(trans.configuration.stack[-2])
                s1elems = self.vocabulary.getIndices(s1Tokens, getPos=getPos, getToken=getToken)
        s0elems = padSequence(s0elems, "s0Padding", emptyIdx)
        s1elems = padSequence(s1elems, "s1Padding", emptyIdx)
        if trans.configuration.buffer:
            bTokens = trans.configuration.buffer[:2]
            belems = self.vocabulary.getIndices(bTokens, getPos=getPos, getToken=getToken)
        belems = padSequence(belems, "bPadding", emptyIdx)
        words = np.concatenate((s0elems, s1elems, belems), axis=0)
        return words

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
            if configuration['model']['inputItems'] == 4:
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
            if configuration['model']['inputItems'] == 4:
                tokenIdxs.append(emptyTokenIdx)
                posIdxs.append(emptyPosIdx)

        return np.asarray(tokenIdxs), np.asarray(posIdxs)


def padSequence(seq, label, emptyIdx):
    padConf = configuration["model"]["padding"]
    return np.asarray(pad_sequences([seq], maxlen=padConf[label], value=emptyIdx))[0]


def initMatrix(vocabulary, usePos=False, useToken=False):
    if usePos:
        if not configuration["model"]["embedding"]["initialisation"]["pos"]:
            return None
        indices = vocabulary.posIndices
        dim = embConf["posEmb"]
        embeddings = vocabulary.posEmbeddings
    elif useToken:
        if not configuration["model"]["embedding"]["initialisation"]["token"]:
            return None
        indices = vocabulary.tokenIndices
        dim = embConf["tokenEmb"]
        embeddings = vocabulary.tokenEmbeddings
    else:
        indices = vocabulary.indices
        dim = vocabulary.embDim
        embeddings = vocabulary.embeddings

    matrix = np.zeros((len(indices), dim))
    for elem in indices:
        matrix[indices[elem]] = embeddings[elem]
    return matrix


def eleminateMarginalClasses(labels, data, taux=5):
    statTuples = sorted(Counter(labels).items())
    print statTuples
    for t in statTuples:
        if t[1] < taux:
            indices = [i for i, x in enumerate(labels) if x == t[0]]
            for idx in list(reversed(indices)):
                labels = numpy.delete(labels, idx)
                data = numpy.delete(data, idx)
    print sorted(Counter(labels).items())
    return labels, data
