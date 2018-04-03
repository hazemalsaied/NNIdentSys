import sys

import numpy

import linarKerasModel as lkm
import newNetwork
import oracle
import v2classification as v2
from corpus import *
from evaluation import evaluate
from network import Network, train
from normalisation import Normalizer
from parser import parse

langs = ['FR']


def identify(loadFolderPath='', load=False):
    configuration["evaluation"]["load"] = load
    for lang in langs:
        corpus = Corpus(lang)
        normalizer, network = parseAndTrain(corpus, loadFolderPath)
        parse(corpus, network, normalizer)
        evaluate(corpus)


def parseAndTrain(corpus, loadFolderPath=''):
    if configuration["evaluation"]["load"]:
        normalizer = reports.loadNormalizer(loadFolderPath)
        logging.warn('Vocabulary size:{0}'.format(len(normalizer.vocabulary.indices)))
        network = Network(normalizer)
        network.model = reports.loadModel(loadFolderPath)
    else:
        if not configuration["evaluation"]["cv"]["active"]:
            reports.createReportFolder(corpus.langName)
        oracle.parse(corpus)
        normalizer = Normalizer(corpus)
        network = Network(normalizer)
        train(network.model, normalizer, corpus)
    return normalizer, network


def crossValidation(debug=False):
    configuration["evaluation"]["cv"]["active"], scores, iterations = True, [0.] * 28, 5
    for lang in langs:
        reports.createReportFolder(lang)
        for cvIdx in range(configuration["evaluation"]["cv"]["currentIter"]):
            reports.createHeader('Iteration no.{0}'.format(cvIdx))
            configuration["evaluation"]["cv"]["currentIter"] = cvIdx
            cvCurrentIterFolder = os.path.join(reports.XP_CURRENT_DIR_PATH,
                                               str(configuration["evaluation"]["currentIter"]))
            if not os.path.isdir(cvCurrentIterFolder):
                os.makedirs(cvCurrentIterFolder)
            corpus = Corpus(lang)
            if debug:
                corpus.trainDataSet = corpus.trainDataSet[:100]
                corpus.testDataSet = corpus.testDataSet[:100]
            testRange, trainRange = corpus.getRangs()
            getTrainAndTestSents(corpus, testRange[cvIdx], trainRange[cvIdx])
            corpus.extractDictionaries()
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
    if len(trainRange) == 2:
        corpus.trainingSents = sent[trainRange[0]:trainRange[1]]
    else:
        corpus.trainingSents = sent[trainRange[0]:trainRange[1]] + sent[trainRange[2]:trainRange[3]]


def identifyLinearKeras():
    evlaConf = configuration["evaluation"]
    evlaConf["cluster"] = True
    evlaConf["debug"] = False
    evlaConf["debugTrainNum"] = 200
    evlaConf["train"] = True
    for lang in langs:
        corpus = Corpus(lang)
        oracle.parse(corpus)
        normalizer = lkm.Normalizer(corpus)
        network = lkm.LinearKerasModel(len(normalizer.tokens) + len(normalizer.pos))
        lkm.train(network.model, corpus, normalizer)
        parse(corpus, network, normalizer)
        evaluate(corpus)


def identifyV2():
    print configuration["features"]
    for lang in langs:
        logging.warn('*' * 20)
        logging.warn('Language: {0}'.format(lang))
        corpus = Corpus(lang)
        oracle.parse(corpus)
        clf, vec = v2.train(corpus)
        v2.parse(corpus, clf, vec)
        evaluate(corpus)
        logging.warn('*' * 20)


def identifyAttached():
    for lang in langs:
        corpus = Corpus(lang)
        oracle.parse(corpus)
        normalizer = Normalizer(corpus)
        network = newNetwork.Network(normalizer)
        newNetwork.train(network.model, normalizer, corpus)
        parse(corpus, network, normalizer)
        evaluate(corpus)


def getScores(newFile, xpNum=5, getTitle=True, getParams=True):
    path ='../Reports/{0}'.format(newFile)
    titles, params, scores = [], [], []
    with open(path, 'r') as log:
        for line in log.readlines():
            paramLine = 'WARNING:root:Parameter number: '
            if getParams and line.startswith(paramLine):
                paramsValue = toNum(line[len(paramLine):len(paramLine) + 8].strip())
                params.append(round(int(paramsValue) / 1000000., 2))
            scoreLine = 'WARNING:root:Ordinary : F-Score: 0'
            if line.startswith(scoreLine):
                fScore = toNum(line[len(scoreLine):len(scoreLine) + 5].strip())
                while len(fScore) < 4:
                    fScore = fScore + '0'
                scores.append(round(int(fScore) / 10000., 4))
            titleLine = 'WARNING:root:Title: '
            if getTitle and line.startswith(titleLine) and not line.startswith('WARNING:root:Title: Language : FR'):
                titles.append(line[len(titleLine):].strip())
    addedItems, newScores, newTitles, newParams = 0, [], [], []
    for i in range(1, len(scores)):
        if addedItems == xpNum:
            addedItems = 0
            continue
        newScores.append(scores[i])
        if getParams:
            newParams.append(params[i])
        if getTitle:
            newTitles.append(titles[i])
        addedItems += 1
    scores = newScores
    if getTitle :
        titles = newTitles
    if getParams:
        params = newParams

    idx, avg, paramAvg, resTxt = 0, 0, 0, ''
    avgTitles, avgScores, avgParams, variances, scorePopulation, bestScores = [], [], [], [], [], []
    for i in range(len(scores) + 1):
        if i != 0 and i % xpNum == 0:
            avgScores.append(round(avg / float(xpNum), 2))
            avgParams.append((round(paramAvg / float(xpNum), 2)))
            bestScores.append(max(scorePopulation))
            if getTitle:
                avgTitles.append(titles[i - 1])
            var = round(numpy.var(scorePopulation) * 100, 2)
            variances.append(var)
            if i != len(scores):
                avg = scores[i]
                if getParams:
                    paramAvg = params[i]
                scorePopulation = [scores[i]]
        else:
            avg += scores[i]
            if getParams:
                paramAvg += params[i]
            scorePopulation.append(scores[i])
    resDetailed = ''
    # for i in range(len(scores)):
    #     resDetailed += str(titles[i]) + ',' + str(scores[i]) + ',' + str(params[i]) + '\n'
    # with open('../Reports/res1.csv', 'w') as res:
    #     res.write(resDetailed)

    for i in range(len(avgScores)):
        if not getParams and not getTitle:
            resTxt += '{0}\t&{1}\t&{2}\\\\\n'.format(avgScores[i], bestScores[i], variances[i])
        elif  not getTitle:
            resTxt += '{0}\t&{1}\t&{2}\t&{3}\t\\\\\n'.format(avgScores[i],bestScores[i], variances[i], avgParams[i])
        elif not getParams:
            resTxt += '{0}\t&\t{1}&\t{2}\t&{3}\\\\\n'.format(avgTitles[i], avgScores[i], bestScores[i],
                                                                   variances[i])
        else:
            resTxt += '{0}\t&\t{1}&\t{2}\t&{3}\t&{4}\\\\\n'.format(avgTitles[i], avgScores[i],bestScores[i], variances[i], avgParams[i])
    with open('../Reports/{0}.csv'.format(newFile), 'w') as res:
        res.write(resTxt)


def toNum(text, addPoint=False):
    textBuf = ''
    for c in text:
        if c.isdigit():
            textBuf += c
    if addPoint and textBuf.strip() != '0':
        return float(textBuf) / (pow(10, len(textBuf)))
    return textBuf


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    getScores('5-StandardXP', xpNum=10, getTitle=False, getParams=False)
