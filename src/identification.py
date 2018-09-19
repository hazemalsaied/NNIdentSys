import datetime
import logging

import torch
from theano import function, config, shared, tensor

import modelCompo
import modelKiperwasser
import modelLinear
import modelLinearKeras as lkm
import modelNonCompo
import modelPytorch
import modelRnn
import modelRnnNonCompo
import oracle
import reports
from corpus import *
from evaluation import evaluate
from normalisation import Normalizer
from parser import parse


def xp(langs, xpNum=3, title='', seed=0):
    verifyGPU()
    evlaConf = configuration['evaluation']
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    reports.createHeader(title)
    if not evlaConf['cv']['active']:
        for lang in langs:
            for i in range(xpNum):
                corpus = Corpus(lang)
                oracle.parse(corpus)
                startTime = datetime.datetime.now()
                network, normalizer = parseAndTrain(corpus)
                sys.stdout.write(reports.doubleSep + reports.tabs + 'Training time : {0}'.
                                 format(datetime.datetime.now() - startTime) + reports.doubleSep)
                parse(corpus, network, normalizer)
                reports.printParsedSents(corpus, 1)
                evaluate(corpus)
    else:
        sys.stdout.write(reports.doubleSep + reports.tabs + 'CV Mode' + reports.doubleSep)
        for i in range(xpNum):
            crossValidation(langs)


def parseAndTrain(corpus):
    if configuration['xp']['linear']:
        res = modelLinear.train(corpus)
        return res
    if configuration['xp']['rnn']:
        network = modelRnn.Network(corpus)
        modelRnn.train(network, corpus)
        return network, None
    if configuration['xp']['rnnNonCompo']:
        network = modelRnnNonCompo.Network(corpus)
        modelRnnNonCompo.train(network, corpus)
        return network, None
    if configuration['xp']['kiperwasser']:
        network = modelKiperwasser.train(corpus)
        normalizer = Normalizer(corpus)
        return network, normalizer
    normalizer = Normalizer(corpus)
    if configuration['xp']['pytorch']:
        network = modelPytorch.PytorchModel(normalizer)
        modelPytorch.main(network, corpus, normalizer)
    elif configuration['xp']['compo']:
        network = modelCompo.Network(normalizer)
        modelCompo.train(network.model, normalizer, corpus)
    else:
        network = modelNonCompo.Network(normalizer)
        modelNonCompo.train(network.model, normalizer, corpus)
    return network, normalizer


def crossValidation(langs, debug=False):
    configuration['evaluation']['cv']['active'], scores, iterations = True, [0.] * 28, 5
    for lang in langs:
        reports.createReportFolder(lang)
        for cvIdx in range(configuration['evaluation']['cv']['currentIter']):
            reports.createHeader('Iteration no.{0}'.format(cvIdx))
            configuration['evaluation']['cv']['currentIter'] = cvIdx
            cvCurrentIterFolder = os.path.join(reports.XP_CURRENT_DIR_PATH,
                                               str(configuration['evaluation']['currentIter']))
            if not os.path.isdir(cvCurrentIterFolder):
                os.makedirs(cvCurrentIterFolder)
            corpus = Corpus(lang)
            if debug:
                corpus.trainDataSet = corpus.trainDataSet[:100]
                corpus.testDataSet = corpus.testDataSet[:100]
            testRange, trainRange = corpus.getRangs()
            getTrainAndTestSents(corpus, testRange[cvIdx], trainRange[cvIdx])
            corpus.extractDictionaries()
            oracle.parse(corpus)
            normalizer, network = parseAndTrain(corpus)
            parse(corpus, network, normalizer)
            tmpScores = evaluate(corpus)
            if len(tmpScores) != len(scores):
                print 'iter scores length are not equal!'
            for i in range(len(tmpScores)):
                if (isinstance(tmpScores[i], float) or isinstance(tmpScores[i], int)) and isinstance(scores[i], float):
                    scores[i] += float(tmpScores[i])
                elif tmpScores[i]:
                    scores[i] = tmpScores[i]
        reports.saveCVScores(scores)


def getTrainAndTestSents(corpus, testRange, trainRange):
    sent = corpus.trainDataSet
    corpus.testingSents = sent[testRange[0]:testRange[1]]
    corpus.trainingSents = sent[trainRange[0]:trainRange[1]] if len(trainRange) == 2 else \
        sent[trainRange[0]:trainRange[1]] + sent[trainRange[2]:trainRange[3]]


def identifyLinearKeras(langs, ):
    for lang in langs:
        corpus = Corpus(lang)
        oracle.parse(corpus)
        normalizer = lkm.Normalizer(corpus)
        network = lkm.LinearKerasModel(len(normalizer.tokens) + len(normalizer.pos))
        lkm.train(network.model, corpus, normalizer)
        parse(corpus, network, normalizer)
        evaluate(corpus)


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


from config import Evaluation, XpMode, Dataset


def setTrainAndTest(v):
    configuration['evaluation'].update({
        'cv': {'active': True if v == Evaluation.cv else False},
        'corpus': True if v == Evaluation.corpus else False,
        'fixedSize': True if v == Evaluation.fixedSize else False,
        'dev': True if v == Evaluation.dev else False,
        'trainVsDev': True if v == Evaluation.trainVsDev else False,
        'trainVsTest': True if v == Evaluation.trainVsTest else False,
    })
    trues, evaluation = 0, ''
    for k in configuration['evaluation'].keys():
        if configuration['evaluation'][k] == True and type(True) == type(configuration['evaluation'][k]):
            evaluation = k.upper()
            trues += 1
    assert trues <= 1, 'There are more than one evaluation settings!'
    sys.stdout.write( tabs + 'Division: {0}'.format(evaluation) + doubleSep)

def setXPMode(v):
    configuration['xp'].update({
        'linear': True if v == XpMode.linear else False,
        'compo': True if v == XpMode.compo else False,
        'pytorch': True if v == XpMode.pytorch else False,
        'kiperwasser': True if v == XpMode.kiperwasser else False,
        'rnn': True if v == XpMode.rnn else False,
        'rnnNonCompo': True if v == XpMode.rnnNonCompo else False,
    })
    trues, mode = 0, ''
    for k in configuration['xp'].keys():
        if configuration['xp'][k] == True and type(True) == type(configuration['xp'][k]):
            mode = k.upper()
            trues += 1
    assert trues <= 1, 'There are more than one experimentation mode!'
    sys.stdout.write( tabs + 'Mode: {0}'.format(mode) + doubleSep)

def setDataSet(v):
    configuration['dataset'].update(
        {
            'sharedtask2': True if v == Dataset.sharedtask2 else False,
            'FTB': True if v == Dataset.FTB else False
        })
    assert configuration['dataset']['sharedtask2'] != configuration['dataset']['FTB'], 'Ambigious data set definition!'
    ds = 'Sharedtask 1.1' if configuration['dataset']['sharedtask2'] else (
        'FTB' if configuration['dataset']['FTB'] else 'Sharedtask 1.0')
    sys.stdout.write( tabs + 'Dataset: {0}'.format(ds) + doubleSep)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    xp(['FR'])
