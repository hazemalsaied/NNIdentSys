import sys

import numpy

import linarKerasModel as lkm
import newNetwork
import oracle
import linearModel
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
        clf, vec = linearModel.train(corpus)
        linearModel.parse(corpus, clf, vec)
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


def mineFile(newFile):
    path = '../Reports/{0}'.format(newFile)
    titles, params, scores = [], [], []
    with open(path, 'r') as log:
        for line in log.readlines():
            paramLine = 'WARNING:root:Parameter number: '
            if line.startswith(paramLine):
                paramsValue = toNum(line[len(paramLine):len(paramLine) + 8].strip())
                params.append(round(int(paramsValue) / 1000000., 2))
            scoreLine = 'WARNING:root:Ordinary : F-Score: 0'
            if line.startswith(scoreLine):
                fScore = toNum(line[len(scoreLine):len(scoreLine) + 5].strip())
                while len(fScore) < 4:
                    fScore = fScore + '0'
                scores.append(round(int(fScore) / 10000., 4) * 100)
            titleLine = 'WARNING:root:Title: '
            if line.startswith(titleLine) and not line.startswith('WARNING:root:Title: Language : FR'):
                titles.append(line[len(titleLine):].strip())
    return titles, scores, params


def clean(titles, scores, params, xpNum):
    addedItems, newScores, newTitles, newParams = 0, [], [], []
    for i in range(1, len(scores)):
        if addedItems == xpNum:
            addedItems = 0
            continue
        newScores.append(scores[i])
        if i < len(params):
            newParams.append(params[i])
        if i < len(titles):
            newTitles.append(titles[i])
        addedItems += 1
    return newTitles, newScores, newParams


def divide(list, subListLength):
    stepNum, newList = len(list) / subListLength, []
    if stepNum:
        for i in range(stepNum):
            newList.append(list[i * subListLength: (i + 1) * subListLength])
        return newList
    return None


def getScores(newFile, xpNum=5):
    titles, scores, params = mineFile(newFile)
    titles, scores, params = clean(titles, scores, params, xpNum)
    getDetailedScores(newFile, scores, titles, params)
    getBrefScores(newFile, scores, titles, params, xpNum)
    resDetailed = ''
    # for i in range(len(scores)):
    #     resDetailed += str(titles[i]) + ',' + str(scores[i]) + ',' + str(params[i]) + '\n'
    # with open('../Reports/res1.csv', 'w') as res:
    #     res.write(resDetailed)


def getDetailedScores(newFile, scores, titles, params):
    text = '\\textbf{title}\t\t&\t\t\\textbf{F}\t\t\\textbf{P}\t\t\\\\\\hline\n'
    for i in range(len(scores)):
        titleText = titles[i] if i < len(titles) else ''
        paramsText = '\t\t&\t\t{0}\t'.format(params[i] if i < len(params) else '')
        paramsText = paramsText if paramsText != '\t\t&\t\t' else ''
        text += '{0}\t\t&\t\t{1}{2}\\\\\n'.format(titleText, scores[i], paramsText)
    with open('../Reports/{0}.detailed.csv'.format(newFile), 'w') as res:
        res.write(text)


def getBrefScores(newFile, scores, titles, params, xpNum):
    scores = divide(scores, xpNum)
    params = divide(params, xpNum)
    titles = divide(titles, xpNum)
    text = '\\textbf{title}\t&\t\\textbf{F$_{mean}$}\t&\t\\textbf{F$_{max}$}\t&' \
           '\t\t\\textbf{MAD}\t\t&\t\t\\textbf{P}\t\t\\\\\\hline\n'
    for i in range(len(scores)):
        titleText = titles[i][0] if titles else ''
        population = scores[i]
        meanValue = round(numpy.mean(population), 1)
        maxValue = round(max(population), 1)
        mad = getMeanAbsoluteDeviation(population)
        paramsText = '\t\t&{0}\t\t'.format(round(numpy.mean(params[i]), 3) if params else '')
        paramsText = paramsText if paramsText != '\t\t&\t\t' else ''
        text += '{0}\t\t&\t\t{1}\t\t&\t\t{2}\t\t&\t\t{3}{4}\t\t\\\\\n'.format(
            titleText, meanValue, maxValue, mad, paramsText)
    with open('../Reports/{0}.csv'.format(newFile), 'w') as res:
        res.write(text)


def toNum(text, addPoint=False):
    textBuf = ''
    for c in text:
        if c.isdigit():
            textBuf += c
    if addPoint and textBuf.strip() != '0':
        return float(textBuf) / (pow(10, len(textBuf)))
    return textBuf



def getMeanAbsoluteDeviation(domain):
    avg = numpy.mean(domain)
    distances = []
    for v in domain:
        dis = v - avg
        if dis < 0:
            dis *= -1
        distances.append(dis)
    return round(sum(distances) / len(distances), 1)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    getScores('8-earlyStoppingErr', xpNum=10)
