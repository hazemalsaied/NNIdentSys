import os

import numpy as np
from numpy import zeros

from config import configuration


def getEmbMatrix(lang, vocab):
    # vocab = list(getCorpusTokens(lang))
    wordEmbDic = getEmbs(lang)
    vocabSize = len(vocab)
    idxs = range(0, vocabSize)
    embeddingMatrix = zeros((vocabSize, configuration['mlp']['tokenEmb']))
    for i in idxs:
        if vocab[i] in wordEmbDic.keys():
            embeddingMatrix[i] = wordEmbDic[vocab[i]]
        else:
            newWord = False
            if '_' in vocab[i]:
                words= vocab[i].split('_')
                avVec =  np.zeros((1, configuration['mlp']['tokenEmb']))
                for w in words:
                    if w in wordEmbDic.keys():
                        avVec += wordEmbDic[w]
                    else:
                        newWord = True
                        break
                if not newWord:
                    embeddingMatrix[i] = np.array(avVec) / len(words)
            if '_' not in vocab[i] or newWord:
                embeddingMatrix[i] = np.random.rand(1, configuration['mlp']['tokenEmb'])

    # embeddingMatrix[-1] = np.random.rand(1, configuration['mlp']['tokenEmb'])
    # embeddingMatrix[-2] = np.random.rand(1, configuration['mlp']['tokenEmb'])
    # embeddingMatrix[-3] = np.random.rand(1, configuration['mlp']['tokenEmb'])
    vocabIdx = dict()
    for i in idxs:
        vocabIdx[vocab[i]] = i
    # vocabIdx[configuration['constants']['unk']] = len(vocab) + 1
    # vocabIdx[configuration['constants']['empty']] = len(vocab) + 1
    # vocabIdx[configuration['constants']['number']] = len(vocab) + 1
    return vocabIdx, embeddingMatrix


def getCorpusTokens(lang):
    tokenDic = set()
    from corpus import Corpus
    corpus = Corpus(lang)
    for s in corpus.trainDataSet + corpus.testDataSet:
        for t in s.tokens:
            tokenDic.add(t.text)
            tokenDic.add(t.lemma)
    return tokenDic


def createLightEmbs(fbFolder, filePath, lang):
    lightEmbstr = ''
    tokenDic = getCorpusTokens(lang)
    with open(os.path.join(fbFolder, filePath), 'r') as f:
        idx = 0
        for l in f:
            if idx == 0:
                idx += 1
            else:
                lineParts = l.split(' ')
                if lineParts[0] in tokenDic:
                    lightEmbstr += l
    with open(os.path.join(fbFolder + 'Light', filename), 'w') as ff:
        ff.write(lightEmbstr)


def getEmbs(lang):
    wordEmb = dict()
    embPath = os.path.join(configuration['path']['projectPath'],
                           'ressources/FacebookEmbsLight/wiki.{0}.vec'.format(lang.lower()))
    with open(embPath, 'r') as f:
        for l in f:
            parts = l.split(' ')
            wordEmb[parts[0].lower()] = parts[1:-1]
    return wordEmb


if __name__ == '__main__':
    # from corpus import *
    from xpTools import setXPMode, setDataSet, setTrainAndTest, Dataset, XpMode, Evaluation

    folder = '/Users/halsaied/PycharmProjects/NNIdenSys/ressources/FacebookEmbs'
    setXPMode(XpMode.linear)
    setTrainAndTest(Evaluation.corpus)
    setDataSet(Dataset.sharedtask2)
    for filename in os.listdir(folder):
        if filename.startswith('wiki'):
            createLightEmbs(folder, filename, filename[5:7].upper())
