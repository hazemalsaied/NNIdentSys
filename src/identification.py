import sys

import numpy

import modelLstmInTwoDirections
import modelMLPLSTM2
import modelMlpHyperSimple
import modelMlpLstm
import modelMlpSimple
import oracle
import v2classification as v2
from corpus import *
from evaluation import evaluate
from parser import parse

# langs = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT', 'LT', 'MT', 'PL', 'PT', 'RO', 'SL', 'SV', 'TR']
langs = ['FR']


def xp(debug=False, train=False,cvDebug=False, cv=False):
    settings.USE_CLUSTER = True
    ######################################
    #   Debug
    ######################################
    if debug:
        settings.XP_DEBUG_DATA_SET = True
        identify()
    ######################################
    #   Train
    ######################################
    if train:
        settings.XP_DEBUG_DATA_SET = False
        settings.XP_TRAIN_DATA_SET = True
        identify()
    ######################################
    #   CV
    ######################################
    if cvDebug:
        settings.XP_DEBUG_DATA_SET = False
        settings.XP_TRAIN_DATA_SET = False
        crossValidation(debug=True)
    ######################################
    #   CV
    ######################################
    if cv:
        settings.XP_DEBUG_DATA_SET = False
        settings.XP_TRAIN_DATA_SET = False
        crossValidation()
        ######################################
        #   Load
        ######################################
        # preTrainedPath= '/home/halsaied/nancy/NNIdenSys/NNIdenSys/Reports/FR-12/12-FR-modelWeigth.hdf5'
        # identify(load=settings.XP_LOAD_MODEL, loadFolderPath=loadFolderPath)


def identify(loadFolderPath='', load=False):
    settings.XP_LOAD_MODEL = load
    for lang in langs:
        corpus = Corpus(lang)
        normalizer, network = parseAndTrain(corpus, loadFolderPath)
        parse(corpus, network, normalizer)
        evaluate(corpus)


def crossValidation(debug=False):
    settings.XP_CROSS_VALIDATION, scores, iterations = True, [0.] * 28, 5
    for lang in langs:
        createReports(lang)
        for cvIdx in range(settings.CV_ITERATIONS):
            reports.createHeader('Iteration no.', cvIdx)
            settings.CV_CURRENT_ITERATION = cvIdx
            cvCurrentIterFolder = os.path.join(reports.XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION))
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


def parseAndTrain(corpus, loadFolderPath=''):
    if settings.XP_LOAD_MODEL:
        normalizer = reports.loadNormalizer(loadFolderPath)
        logging.warn('Vocabulary size:{0}'.format(len(normalizer.vocabulary.indices)))
        network = initNetword(normalizer)
        network.model = reports.loadModel(loadFolderPath)
    else:
        if not settings.XP_CROSS_VALIDATION:
            createReports(corpus.langName)
        oracle.parse(corpus)
        normalizer = initNormaliser(corpus)
        network = initNetword(normalizer)
        network.train(normalizer, corpus)
    return normalizer, network


def initNormaliser(corpus):
    if settings.USE_MODEL_LSTM_IN_TWO_DIRS:
        normalizer = modelLstmInTwoDirections.NormalizerLstmInTwoDirections(corpus)
        return normalizer
    if settings.USE_MODEL_MLP_Hyper_SIMPLE:
        normalizer = modelMlpHyperSimple.NormalizerMLPHyperSimple(corpus)
        return normalizer
    if settings.USE_MODEL_MLP_SIMPLE:
        normalizer = modelMlpSimple.NormalizerMLPSimple(corpus)
        return normalizer
    if settings.USE_MODEL_MLP_LSTM:
        normalizer = modelMlpLstm.NormalizerMlpLstm(corpus)
        return normalizer
    if settings.USE_MODEL_MLP_LSTM_2:
        normalizer = modelMLPLSTM2.NormalizerMlpLstm2(corpus)
        return normalizer


def initNetword(normalizer):
    if settings.USE_MODEL_LSTM_IN_TWO_DIRS:
        network = modelLstmInTwoDirections.NetworkLstmInTwoDirections(normalizer)
        return network
    if settings.USE_MODEL_MLP_Hyper_SIMPLE:
        network = modelMlpHyperSimple.NetworkMLPHyperSimple(normalizer)
        return network
    if settings.USE_MODEL_MLP_SIMPLE:
        network = modelMlpSimple.NetworkMLPSimple(normalizer)
        return network
    if settings.USE_MODEL_MLP_LSTM:
        network = modelMlpLstm.NetworkMlpLstm(normalizer)
        return network
    if settings.USE_MODEL_MLP_LSTM_2:
        network = modelMLPLSTM2.NetworkMlpLstm2(normalizer)
        return network


def analyzeCorporaAndOracle():
    header = 'Non recognizable,Interleaving,Embedded,Distributed Embedded,Left Embedded,Right Embedded,Middle Embedded'
    analysisReport = header + '\n'
    for lang in langs:
        logging.warn('*' * 20)
        logging.warn('Language: {0}'.format(lang))
        corpus = Corpus(lang)
        analysisReport += corpus.getVMWEReport() + '\n'
        oracle.parse(corpus)
        oracle.validate(corpus)
    with open('../Results/VMWE.Analysis.csv', 'w') as f:
        f.write(analysisReport)


def identifyV2():
    for lang in langs:
        logging.warn('*' * 20)
        logging.warn('Language: {0}'.format(lang))
        corpus = Corpus(lang)
        oracle.parse(corpus)
        clf, vec = v2.train(corpus)
        v2.parse(corpus, clf, vec)
        evaluate(corpus)
        logging.warn('*' * 20)


def createReports(lang):
    cpNum = getXPNum()
    if settings.XP_SAVE_MODEL:
        reports.getXPDirectory(lang, cpNum)
        if not settings.XP_LOAD_MODEL:
            logging.warn('Result folder: {0}'.format(reports.XP_CURRENT_DIR_PATH.split('/')[-1]))
            reports.createXPDirectory()


def getXPNum():
    with open('config.text', 'r+') as f:
        content = f.read()
        xpNum = int(content[-3:])
    with open('config.text', 'w') as f:
        newXpNum = xpNum + 1
        if newXpNum < 10:
            newXpNum = '00' + str(newXpNum)
        elif newXpNum < 100:
            newXpNum = '0' + str(newXpNum)
        f.write(content[:-3] + newXpNum)
    return xpNum


reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.WARNING)
numpy.random.seed(7)

settings.USE_MODEL_MLP_SIMPLE = True

reports.createHeader('', 'MLP Simle 5 4 3')
xp(train=True)

reports.createHeader('', 'MLP Simle 2048 + 5 4 3 ')
settings.MLP_LAYER_1_UNIT_NUM = 2048
settings.MLP_LAYER_2_UNIT_NUM = 1024
xp(train=True)
