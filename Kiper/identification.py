import datetime
import logging
import random

import torch
from theano import function, config, shared, tensor

import modelKiperwasser
import oracle
import reports
from corpus import *
from evaluation import evaluate
from parser import parse


def xp(langs, xpNum=3, title='', seed=0):
    verifyGPU()
    evlaConf = configuration['evaluation']
    evlaConf['cluster'] = True
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    reports.createHeader(title)
    reports.printMode()
    for lang in langs:
        for i in range(xpNum):
            corpus = Corpus(lang)
            oracle.parse(corpus)
            startTime = datetime.datetime.now()
            network = modelKiperwasser.train(corpus)
            sys.stdout.write(reports.doubleSep + reports.tabs + 'Training time : {0}'.
                             format(datetime.datetime.now() - startTime) + reports.doubleSep)
            parse(corpus, network)
            evaluate(corpus)


def getTrainAndTestSents(corpus, testRange, trainRange):
    sent = corpus.trainDataSet
    corpus.testingSents = sent[testRange[0]:testRange[1]]
    corpus.trainingSents = sent[trainRange[0]:trainRange[1]] if len(trainRange) == 2 else \
        sent[trainRange[0]:trainRange[1]] + sent[trainRange[2]:trainRange[3]]


def getAllLangStats(langs):
    res = ''
    for lang in langs:
        corpus = Corpus(lang)
        res += corpus.langName + ',' + getStats(corpus.trainDataSet, asCSV=True) + ',' + \
               getStats(corpus.devDataSet, asCSV=True) + ',' + \
               getStats(corpus.testDataSet, asCSV=True) + '\n'
    return res


def analyzeCorporaAndOracle(langs):
    header = 'Non recognizable,Interleaving,Embedded,Distributed Embedded,Left Embedded,Right Embedded,Middle Embedded'
    analysisReport = header + '\n'
    for lang in langs:
        sys.stdout.write('Language = {0}\n'.format(lang))
        corpus = Corpus(lang)
        analysisReport += corpus.getVMWEReport() + '\n'
        oracle.parse(corpus)
        oracle.validate(corpus)
    with open('../Results/VMWE.Analysis.csv', 'w') as f:
        f.write(analysisReport)


def verifyGPU():
    vlen = 10 * 30 * 768
    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], tensor.exp(x))
    if numpy.any([isinstance(x.op, tensor.Elemwise) and
                  ('Gpu' not in type(x.op).__name__)
                  for x in f.maker.fgraph.toposort()]):
        sys.stdout.write(tabs + 'Attention: CPU used\n')
    else:
        sys.stdout.write(tabs + 'GPU Enabled')


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    configuration['xp']['kiperwasser'] = True

    configuration['kiperwasser'] = {
        'wordDim': 25,  # 100,
        'posDim': 5,  # 25,
        'layerNum': 2,
        'activation': 'tanh',
        'optimizer': 'adam',
        'lr': 0.1,
        'dropout': .3,
        'epochs': 40,
        'batch': 1,
        'dense1': 25,  # 100,
        'dense2': 0,
        'denseDropout': False,
        'lstmDropout': 0.25,
        'lstmLayerNum': 2,
        'focusedElemNum': 8,
        'lstmUnitNum': 8  # 125
    }
    configuration["sampling"]["importantSentences"] = True
    # configuration["evaluation"]["fixedSize"] = True
    configuration["evaluation"]["debugTrainNum"] = 10

    sys.stdout.write(str(configuration['kiperwasser']))

    xp(['FR'], xpNum=1)
