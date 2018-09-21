import random
import sys

import numpy as np
import sklearn.utils
from imblearn.over_sampling import RandomOverSampler

import reports
from config import configuration


def overSample(labels, data):
    tokenData, posData = [], []
    if not configuration["sampling"]["overSampling"]:
        for item in data:
            tokenData.append(np.asarray(item[:configuration['model']['inputItems']]))
            posData.append(np.asarray(item[configuration['model']['inputItems']:]))
        return np.asarray(labels), [np.asarray(tokenData), np.asarray(posData)]
    sys.stdout.write(reports.tabs + 'data size before sampling = {0}\n'.format(len(labels)))
    ros = RandomOverSampler(random_state=0)
    data, labels = ros.fit_sample(data, labels)
    for item in data:
        tokenData.append(np.asarray(item[:configuration['model']['inputItems']]))
        posData.append(np.asarray(item[configuration['model']['inputItems']:]))
    sys.stdout.write(reports.tabs + 'data size after sampling = {0}\n'.format(len(labels)))
    return np.asarray(labels), [np.asarray(tokenData), np.asarray(posData)]


def overSampleImporTrans(data, labels, corpus, normalizer):
    tokenData, newLabels = [], []
    traitedMWEs = set()
    oversamplingTaux = configuration["sampling"]["mweRepeition"]
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


def getClassWeights(labels):
    if not configuration['sampling']['sampleWeight']:
        return {}
    classes = np.unique(labels)
    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes, labels)
    res = dict()
    for c in classes:
        cIdx = classes.tolist().index(c)
        res[c] = float(class_weight[cIdx] * configuration["sampling"]["favorisationCoeff"]) if c > 1 else class_weight[cIdx]
    sys.stdout.write(reports.tabs + 'Favorisation Coeff : {0}\n'.format(configuration["sampling"]["favorisationCoeff"]))

    return res


def shuffle(lists):
    newLists = []
    for i in range(len(lists)):
        newLists.append([])
    r = range(lists[0].shape[0])
    random.shuffle(r)
    for i in r:
        for j in range(len(lists)):
            newLists[j].append(lists[j][i])
    return newLists


def getSampleWeightArray(labels, classWeightDic):
    if not configuration['sampling']['sampleWeight']:
        return None
    sampleWeights = []
    for l in labels:
        sampleWeights.append(classWeightDic[l])
    return np.asarray(sampleWeights)


def getMWEDicAsIdxs(corpus, normalizer):
    result = dict()
    for mwe in corpus.mweDictionary:
        idx = getIdxsStrForMWE(mwe, normalizer)
        if idx:
            result[idx] = mwe
    return result


def getIdxsStrForMWE(mwe, normalizer):
    tokenLemmas = mwe.replace(' ', '_')
    if tokenLemmas not in normalizer.vocabulary.tokenIndices:
        return None
    return normalizer.vocabulary.tokenIndices[tokenLemmas]


def shuffleTwoArrayInParallel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    rangee = range(a.shape[0])
    random.shuffle(rangee)
    if b.shape[0] == 2:
        aTmp, bTmp, cTmp = [], [], []
        for i in rangee:
            aTmp.append(a[i])
            bTmp.append(b[0][i])
            cTmp.append(b[1][i])
        return np.asarray(aTmp), [np.asarray(bTmp), np.asarray(cTmp)]
    elif b.shape[1] == configuration['model']['inputItems'] * 2:
        aTmp, bTmp = [], []
        for i in rangee:
            aTmp.append(a[i])
            bTmp.append(b[i])
        return np.asarray(aTmp), np.asarray(bTmp)
    assert 'Irrelevent array size!'


def test():
    a = [1, 2, 3, 4, 5, 6]
    b = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    print shuffleTwoArrayInParallel(np.asarray(a), np.asarray(b))
