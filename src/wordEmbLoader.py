import gensim
import numpy as np

from config import configuration

unk = configuration["constants"]["unk"]
empty = configuration["constants"]["empty"]
number = configuration["constants"]["number"]


def initialisePOS(corpus, indices):
    embConf = configuration["model"]["embedding"]
    embeddings, idx = dict(), 0
    traindEmb = trainPosEmbWithWordToVec(corpus)
    for elem in indices:
        if elem.lower() in traindEmb.wv.vocab:
            embeddings[elem.lower()] = traindEmb.wv[elem]
        else:
            embeddings[elem.lower()] = getRandomVector(embConf["posEmb"])
    # for elem in traindEmb.wv.vocab:
    #     if elem.lower() in indices and indices[elem] > 1:
    #         embeddings[elem.lower()] = traindEmb.wv[elem]
    embeddings[unk] = getRandomVector(embConf["posEmb"])
    if not embConf["concatenation"]:
        embeddings[empty] = getRandomVector(embConf["posEmb"])
    return embeddings


def trainPosEmbWithWordToVec(corpus):
    embConf = configuration["model"]["embedding"]
    initConf = configuration["model"]["embedding"]["initialisation"]
    normailsedSents = []
    for sent in corpus.trainingSents + corpus.testingSents:
        normailsedSent = []
        for token in sent.tokens:
            normailsedSent.append(token.posTag.lower())
        normailsedSents.append(normailsedSent)
    model = gensim.models.Word2Vec(normailsedSents, size=embConf["posEmb"], window=initConf["Word2VecWindow"])
    return model


def getFreqDic(corpus, posTag=False):
    freqDic = dict()
    for sent in corpus.trainingSents:
        for token in sent.tokens:
            key = token.posTag.lower() if posTag else token.getTokenOrLemma().lower()
            if key not in freqDic:
                freqDic[key] = 1
            else:
                freqDic[key] += 1
    return freqDic


def getRandomVector(length):
    return np.asarray([float(val) for val in np.random.uniform(low=-0.01, high=0.01, size=int(length))])
