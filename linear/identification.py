import datetime
import logging

import modelLinear
import oracle
import reports
from corpus import *
from evaluation import evaluate
from parser import parse


def identify(lang):
    corpus = Corpus(lang)
    oracle.parse(corpus)
    startTime = datetime.datetime.now()
    network, normalizer = parseAndTrain(corpus)
    sys.stdout.write(reports.doubleSep + reports.tabs + 'Training time : {0}'.
                     format(datetime.datetime.now() - startTime) + reports.doubleSep)
    parse(corpus, network, normalizer)
    reports.printParsedSents(corpus, 1)
    evaluate(corpus)


def parseAndTrain(corpus):
    if configuration['xp']['linear']:
        res = modelLinear.train(corpus)
        return res


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
    sys.stdout.write(tabs + 'Division: {0}'.format(evaluation) + doubleSep)


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
    sys.stdout.write(tabs + 'Mode: {0}'.format(mode) + doubleSep)


def setDataSet(v):
    configuration['dataset'].update(
        {
            'sharedtask2': True if v == Dataset.sharedtask2 else False,
            'FTB': True if v == Dataset.FTB else False
        })
    assert configuration['dataset']['sharedtask2'] != configuration['dataset']['FTB'], 'Ambigious data set definition!'
    ds = 'Sharedtask 1.1' if configuration['dataset']['sharedtask2'] else (
        'FTB' if configuration['dataset']['FTB'] else 'Sharedtask 1.0')
    sys.stdout.write(tabs + 'Dataset: {0}'.format(ds) + doubleSep)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    xp(['FR'])
