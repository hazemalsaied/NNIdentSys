import sys

import numpy

import modelConfTrans
import modelLstmInTwoDirections
import modelMLPLSTM2
import modelMlpAuxSimple
import modelMlpHyperSimple
import modelMlpLstm
import modelMlpSimple
import oracle
import v2classification as v2
from corpus import *
from evaluation import evaluate
from parser import parse

allLangs = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT', 'LT', 'MT', 'PL', 'PT', 'RO', 'SL', 'SV', 'TR']
langs = ['FR']


def xp(train=False, cv=False, xpNum=1):
    settings.USE_CLUSTER = True
    if train:
        ######################################
        #   Debug
        ######################################
        settings.XP_DEBUG_DATA_SET = True
        identify()
        ######################################
        #   Train
        ######################################
        settings.XP_DEBUG_DATA_SET = False
        settings.XP_TRAIN_DATA_SET = True
        for i in range(xpNum):
            identify()
        settings.XP_TRAIN_DATA_SET = False
    if cv:
        ######################################
        #   CV Debug
        ######################################
        crossValidation(debug=True)
        ######################################
        #   CV
        ######################################
        for i in range(xpNum):
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
        reports.createReportFolder(lang)
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
            reports.createReportFolder(corpus.langName)
        oracle.parse(corpus)
        normalizer = initNormaliser(corpus)
        network = initNetword(normalizer)
        network.train(normalizer, corpus)
    return normalizer, network


def initNormaliser(corpus):
    if settings.USE_MODEL_CONF_TRANS:
        normalizer = modelConfTrans.NormalizerConfTrans(corpus)
        return normalizer
    if settings.USE_MODEL_LSTM_IN_TWO_DIRS:
        normalizer = modelLstmInTwoDirections.NormalizerLstmInTwoDirections(corpus)
        return normalizer
    if settings.USE_MODEL_MLP_AUX_SIMPLE:
        normalizer = modelMlpAuxSimple.NormalizerMLPAuxSimple(corpus)
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
    raise ValueError('initNormaliser: No model Selected!')


def initNetword(normalizer):
    if settings.USE_MODEL_CONF_TRANS:
        network = modelConfTrans.NetworkConfTrans(normalizer)
        return network
    if settings.USE_MODEL_MLP_AUX_SIMPLE:
        network = modelMlpAuxSimple.NetworkMLPAuxSimple(normalizer)
        return network
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
    raise ValueError('initNetword: No model Selected!')

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


def xpGRU(moreUnits=False, stacked=False, gru=False, cv=False):
    settings.USE_MODEL_MLP_LSTM = True
    if moreUnits:
        settings.LSTM_1_UNIT_NUM = 1024
        settings.LSTM_2_UNIT_NUM = 1024
    settings.STACKED = stacked
    settings.USE_GRU = gru
    title = ''
    title += '' if not stacked else 'Stacked '
    title += ' LSTM ' if not gru else ' GRU '
    title += ' Expanded Unit Num ' if moreUnits else ''
    reports.createHeader('', title)
    if cv:
        xp(cv=True)
    else:
        xp(train=True)

    settings.LSTM_1_UNIT_NUM = 512
    settings.LSTM_2_UNIT_NUM = 512
    settings.USE_MODEL_MLP_LSTM = False
    settings.STACKED = False
    settings.USE_GRU = False


def xpMLPAux(cv=False):
    settings.USE_MODEL_MLP_AUX_SIMPLE = True
    reports.createHeader('', 'MLP Aux')
    if cv:
        xp(cv=True)
    else:
        xp(train=True)


def xpMLPTotal(cv=False, train=False, xpNum=5):
    settings.USE_MODEL_MLP_SIMPLE = True

    reports.createHeader('', 'Standard')
    xp(cv=cv, train=train, xpNum=xpNum)

    reports.createHeader('', 'Optimizer = ADAM ')
    settings.USE_ADAM = True
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.USE_ADAM = False

    reports.createHeader('', 'Dense 2 not activated')
    settings.USE_DENSE_2 = False
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.USE_DENSE_2 = True

    reports.createHeader('', 'Dense 3 activated')
    settings.USE_DENSE_3 = True
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.USE_DENSE_3 = False

    reports.createHeader('', 'Tahn is activated')
    settings.MLP_USE_TANH_1 = True
    settings.MLP_USE_TANH_2 = True
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.MLP_USE_TANH_1 = False
    settings.MLP_USE_TANH_2 = False

    reports.createHeader('', 'Without weight matrix')
    settings.INITIALIZE_EMBEDDING = False
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.INITIALIZE_EMBEDDING = True

    reports.createHeader('', 'Maximized weight matrix')
    settings.REMOVE_NON_FREQUENT_WORDS = False
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.REMOVE_NON_FREQUENT_WORDS = True

    reports.createHeader('', 'Batch size = 64')
    settings.NN_BATCH_SIZE = 64
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.NN_BATCH_SIZE = 128

    reports.createHeader('', 'Batch size = 256')
    settings.NN_BATCH_SIZE = 256
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.NN_BATCH_SIZE = 128

    reports.createHeader('', 'No early stop')
    settings.EARLY_STOP = False
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.EARLY_STOP = True

    reports.createHeader('', 'Use Lemma')
    settings.M1_USE_TOKEN = False
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.M1_USE_TOKEN = True

    reports.createHeader('', 'No POS emb')
    settings.USE_POS_EMB = False
    xp(cv=cv, train=train, xpNum=xpNum)
    settings.USE_POS_EMB = True


def xpMLP(_543=False, _2048_1024=False, cv=False):
    settings.USE_MODEL_MLP_SIMPLE = True

    title = 'MLP '
    title += '' if not _543 else '5, 4, 3 '
    title += '' if not _2048_1024 else ' 2048 1024 '
    reports.createHeader('', title)

    if _2048_1024:
        settings.MLP_LAYER_1_UNIT_NUM = 2048
        settings.MLP_LAYER_2_UNIT_NUM = 1024
    if cv:
        xp(cv=True)
    else:
        xp(train=True)

    if _2048_1024:
        settings.USE_MODEL_MLP_SIMPLE = False
        settings.MLP_LAYER_1_UNIT_NUM = 1024
        settings.MLP_LAYER_2_UNIT_NUM = 512


def xpConfTrans(stacked=False, gru=False, cv=False):
    settings.USE_MODEL_CONF_TRANS = True
    settings.STACKED = stacked
    settings.USE_GRU = gru
    title = 'Conf <=> Trans: '
    title += '' if not stacked else 'Stacked '
    title += ' LSTM ' if not gru else ' GRU '
    reports.createHeader('', title)
    if cv:
        xp(cv=True)
    else:
        xp(train=True)

    settings.USE_MODEL_CONF_TRANS = False
    settings.STACKED = False
    settings.USE_GRU = False


reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.WARNING)
numpy.random.seed(7)

#xpMLPTotal(cv=True)
settings.USE_MODEL_MLP_SIMPLE = True
xp(train=True)
# xpMLP()
# xpGRU(gru=True, stacked=True, moreUnits=True)
# xpMLPAux()
# xpConfTrans(gru=True, stacked=True)
