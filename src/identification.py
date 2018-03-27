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


def getScores(path):
    titles, params, scores = [], [], []
    with open(path, 'r') as log:
        for line in log.readlines():
            paramLine = 'WARNING:root:Parameter number: '
            if line.startswith(paramLine):
                params.append(line[len(paramLine):len(paramLine) + 8].strip())
            scoreLine = 'WARNING:root:Ordinary : F-Score: '
            if line.startswith(scoreLine):
                scores.append(line[len(scoreLine):len(scoreLine) + 5].strip())
            titleLine = 'WARNING:root:Title: '
            if line.startswith(titleLine):
                titles.append(line[len(titleLine):len(scoreLine) + 5].strip())
    resTxt = ''
    for i in range(len(scores)):
        resTxt += str(scores[i]) + ',' + str(params[i]) + '\n'
    with open('../Results/res.csv', 'w') as res:
        res.write(resTxt)
