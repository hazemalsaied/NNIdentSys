from collections import Counter

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing.sequence import pad_sequences

import reports
from corpus import getTokens
from extraction import Extractor
from reports import *
from vocabulary import Vocabulary, empty


class Normalizer:
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        self.inputListDimension = 2
        global embConf, initConf
        embConf = configuration["model"]["embedding"]
        initConf = embConf["initialisation"]
        self.nnExtractor, self.vocabulary, self.posWeightMatrix, self.weightMatrix, self.tokenWeightMatrix = \
            None, None, None, None, None
        if embConf["active"]:
            self.vocabulary = Vocabulary(corpus)
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
        if configuration["features"]["active"]:
            self.nnExtractor = Extractor(corpus)
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
        imporSents = len(corpus.trainingSents)
        labels, tokenData, posData, featureData, data = [], [], [], [], []
        useFeatures = configuration["features"]["active"]
        for sent in corpus.trainingSents:
            if (configuration["sampling"]["importantTransitions"] or
                configuration["sampling"]["importantSentences"]) and not sent.vMWEs:
                imporSents -= 1
                continue
            trans = sent.initialTransition
            while trans and trans.next:
                if configuration["sampling"]["importantTransitions"] and not trans.isImportant():
                    trans = trans.next
                else:
                    tokenIdxs, posIdxs = self.getAttachedIndices(trans)
                    tokenData.append(np.asarray(tokenIdxs))
                    posData.append(np.asarray(posIdxs))
                    # TODO generalise
                    data.append(np.asarray(np.concatenate((tokenIdxs, posIdxs))))
                    if useFeatures:
                        features = np.asarray(self.nnExtractor.vectorize(trans))
                        featureData.append(np.asarray(features))
                    labels = np.append(labels, trans.next.type.value)
                    trans = trans.next

        sys.stdout.write(reports.seperator + reports.tabs + 'Sampling' + reports.doubleSep)
        sys.stdout.write(reports.tabs + '{0} importatnt sents of {1}\n'.
                         format(imporSents, len(corpus.trainingSents)))
        if useFeatures:
            if configuration["model"]["embedding"]["usePos"]:
                return np.asarray(labels), [np.asarray(tokenData), np.asarray(posData), np.asarray(featureData)]
            else:
                return np.asarray(labels), [np.asarray(tokenData), np.asarray(featureData)]
        if configuration["model"]["embedding"]["usePos"]:
            # labels, data = eleminateMarginalClasses(labels, data)
            self.inputListDimension = 2
            # labels, data = overSample(data, labels)

            # labels, data = shuffleTwoArrayInParallel(labels, [np.asarray(tokenData), np.asarray(posData)])
            # data, labels = ros.fit_sample(np.asarray([np.asarray(tokenData), np.asarray(posData)]),
            # np.asarray(labels))
            if configuration["sampling"]["focused"]:
                data, labels = overSampleImporTrans(data, labels, corpus, self)
            return overSample(data, labels)
            # return overSample(data, labels)
            # np.asarray(labels), [np.asarray(tokenData), np.asarray(posData)]  # labels, data #
            # return np.asarray(labels), [np.asarray(tokenData), np.asarray(posData)]
        else:
            self.inputListDimension = 1
            return np.asarray(labels), np.asarray(tokenData)

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
        emptyTokenIdx = self.vocabulary.attachedTokens[empty]
        emptyPosIdx = self.vocabulary.attachedPos[empty]
        tokenIdxs, posIdxs = [], []
        if trans.configuration.stack:
            s0Tokens = getTokens(trans.configuration.stack[-1])
            tokenIdx, posIdx = self.vocabulary.getAttachedIndices(s0Tokens)
            tokenIdxs.append(tokenIdx)
            posIdxs.append(posIdx)
            if len(trans.configuration.stack) > 1:
                s1Tokens = getTokens(trans.configuration.stack[-2])
                tokenIdx, posIdx = self.vocabulary.getAttachedIndices(s1Tokens)
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
            tokenIdx, posIdx = self.vocabulary.getAttachedIndices([trans.configuration.buffer[0]])
            tokenIdxs.append(tokenIdx)
            posIdxs.append(posIdx)
            if len(trans.configuration.buffer) > 1:
                tokenIdx, posIdx = self.vocabulary.getAttachedIndices([trans.configuration.buffer[1]])
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

        return np.asarray(tokenIdxs), np.asarray(posIdxs)


def overSample(data, labels):
    tokenData, posData = [], []
    if not configuration["sampling"]["overSampling"]:
        for item in data:
            tokenData.append(np.asarray(item[:4]))
            posData.append(np.asarray(item[4:]))
        return np.asarray(labels), [np.asarray(tokenData), np.asarray(posData)]
    sys.stdout.write(reports.tabs + 'data size before sampling = {0}\n'.format(len(labels)))
    ros = RandomOverSampler(random_state=0)
    data, labels = ros.fit_sample(data, labels)
    for item in data:
        tokenData.append(np.asarray(item[:4]))
        posData.append(np.asarray(item[4:]))
    sys.stdout.write(reports.tabs + 'data size after sampling = {0}\n'.format(len(labels)))
    labelsDic = dict()
    for item in labels:
        if item not in labelsDic:
            labelsDic[item] = 1
        else:
            labelsDic[item] += 1
    print labelsDic
    return np.asarray(labels), [np.asarray(tokenData), np.asarray(posData)]


def getMWEDicAsIdxs(corpus, normalizer):
    result = dict()
    for mwe in corpus.mweDictionary:
        result[getIdxsStrForMWE(mwe, normalizer)] = mwe
    return result


def getIdxsStrForMWE(mwe, normalizer):
    tokenLemmas = mwe.replace(' ', '_')
    if tokenLemmas in normalizer.vocabulary.tokenIndices:
        return normalizer.vocabulary.tokenIndices[tokenLemmas]
    else:
        assert 'Something went wrong while transforming a MWE to Idx String'
    # idxs = ''
    # for tokenLemma in tokenLemmas:
    #     if tokenLemma in normalizer.vocabulary.tokenIndices:
    #         idxs += str(normalizer.vocabulary.tokenIndices[tokenLemma]) + '.'
    #     else:
    #         assert 'Something went wrong while transforming a MWE to Idx String'
    # return idxs[:-1]


def overSampleImporTrans(data, labels, corpus, normalizer, oversamplingTaux=configuration["sampling"]["mweRepeition"]):
    tokenData, newLabels = [], []
    traitedMWEs = set()
    mWEDicAsIdxs = getMWEDicAsIdxs(corpus, normalizer)
    for i in range(len(labels)):
        if labels[i] > 2:
            mweIdx = data[i][0]
            if mweIdx in mWEDicAsIdxs and mweIdx not in traitedMWEs:
                traitedMWEs.add(mweIdx)
                mwe = mWEDicAsIdxs[mweIdx]
                mweLength = len(mwe.split(' '))
                mweOccurrence = corpus.mweDictionary[mwe]
                if mweOccurrence < oversamplingTaux:
                    for underProcessingTransIdx in range(i - (2 * mweLength - 1) + 1, i + 1):
                        for j in range(oversamplingTaux - mweOccurrence):
                            tokenData.append(data[underProcessingTransIdx])
                            newLabels.append(labels[underProcessingTransIdx])
    sys.stdout.write(reports.tabs + 'data size before focused sampling = {0}\n'.format(len(labels)))
    labels = np.concatenate((labels, newLabels))
    sys.stdout.write(reports.tabs + 'data size after focused sampling = {0}\n'.format(len(labels)))
    data = np.concatenate((data, tokenData))
    return np.asarray(data), np.asarray(labels)


def getStackLemmaString(stack):
    if not stack:
        return None
    return ' '.join(t.getLemmaOrToken() for t in getTokens(stack[-1]))


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


def shuffleTwoArrayInParallel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    print a.shape, a.size, b.shape, b.size
    assert b.shape[0] == 2
    rangee = range(a.shape[0])
    random.shuffle(rangee)
    aTmp, bTmp, cTmp = [], [], []
    for i in rangee:
        aTmp.append(a[i])
        bTmp.append(b[0][i])
        cTmp.append(b[1][i])
    return np.asarray(aTmp), [np.asarray(bTmp), np.asarray(cTmp)]


def test():
    a = [1, 2, 3, 4, 5, 6]
    b = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    print shuffleTwoArrayInParallel(np.asarray(a), np.asarray(b))


if __name__ == '__main__':
    test()
