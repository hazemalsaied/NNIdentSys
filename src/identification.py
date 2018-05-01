import linarKerasModel as lkm
import linearModel
import newNetwork
import oracle
from corpus import *
from evaluation import evaluate
from network import Network, train
from normalisation import Normalizer
from parser import parse


def identify(langs=['FR'], loadFolderPath='', load=False):
    sys.stdout.write('*' * 20 + '\n')
    sys.stdout.write('Deep model(Padding) \n')
    configuration["evaluation"]["load"] = load
    for lang in langs:
        corpus = Corpus(lang)
        normalizer, network = parseAndTrain(corpus, loadFolderPath)
        parse(corpus, network, normalizer)
        evaluate(corpus)
        sys.stdout.write('*' * 20 + '\n')


def parseAndTrain(corpus, loadFolderPath=''):
    if configuration["evaluation"]["load"]:
        normalizer = reports.loadNormalizer(loadFolderPath)
        network = Network(normalizer)
        network.model = reports.loadModel(loadFolderPath)
    else:
        if not configuration["evaluation"]["cv"]["active"]:
            reports.createReportFolder(corpus.langName)
        oracle.parse(corpus)
        normalizer = Normalizer(corpus)
        printReport(normalizer)
        network = Network(normalizer)
        train(network.model, normalizer, corpus)
    return normalizer, network


def printReport(normalizer):
    sys.stdout.write('# Padding = {0}\n'.format(configuration["model"]["padding"]))
    embConf = configuration["model"]["embedding"]
    sys.stdout.write('# Embedding = {0}\n'.format(embConf["active"]))
    if embConf["active"]:
        sys.stdout.write('# Initialisation = {0}\n'.format(embConf["initialisation"]["active"]))
        sys.stdout.write('# Concatenation = {0}\n'.format(embConf["concatenation"]))
        if embConf["concatenation"]:
            sys.stdout.write('# Emb = {0}\n'.format(embConf["posEmb"] + embConf["tokenEmb"]))
        else:
            sys.stdout.write('# Lemma  = {0}\n'.format(embConf["lemma"]))
            sys.stdout.write('# Token/Lemma emb = {0}\n'.format(embConf["tokenEmb"]))
            sys.stdout.write('# POS = {0}\n'.format(embConf["usePos"]))
            if embConf["usePos"]:
                sys.stdout.write('# POS emb = {0}\n'.format(embConf["posEmb"]))
    sys.stdout.write('# Features = {0}\n'.format(configuration["features"]["active"]))
    if normalizer.nnExtractor:
        sys.stdout.write('# Features = {0}\n'.format(normalizer.nnExtractor.featureNum))


def crossValidation(langs=['FR'], debug=False):
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


def identifyLinearKeras(langs=['FR'], ):
    sys.stdout.write('*' * 20 + '\n')
    sys.stdout.write('Linear model in KERAS\n')
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
        sys.stdout.write('*' * 20 + '\n')


def identifyV2(langs=['FR'] ):
    sys.stdout.write('*' * 20 + '\n')
    sys.stdout.write('Linear model\n')
    print configuration["features"]
    for lang in langs:
        corpus = Corpus(lang)
        oracle.parse(corpus)
        clf, vec = linearModel.train(corpus)
        linearModel.parse(corpus, clf, vec)
        evaluate(corpus)
        sys.stdout.write('*' * 20 + '\n')


def identifyAttached(lang='FR'):
    sys.stdout.write('*' * 20 + '\n')
    sys.stdout.write('Deep model(No padding)\n')
    corpus = Corpus(lang)
    oracle.parse(corpus)
    normalizer = Normalizer(corpus)
    printReport(normalizer)
    network = newNetwork.Network(normalizer)
    newNetwork.train(network.model, normalizer, corpus)
    parse(corpus, network, normalizer)
    evaluate(corpus)
    sys.stdout.write('*' * 20 + '\n')


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    # getScores('8-earlyStoppingErr', xpNum=10)
