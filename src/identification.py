import random
import sys

import numpy

import linarKerasModel as lkm
import oracle
import v2classification as v2
from corpus import *
from evaluation import evaluate
from network import Network, train
from normalisation import Normalizer
from parser import parse

allLangs = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT', 'LT', 'MT', 'PL', 'PT', 'RO', 'SL', 'SV', 'TR']
langs = ['FR']


def identify(loadFolderPath='', load=False):
    configuration["evaluation"]["load"] = load
    for lang in langs:
        corpus = Corpus(lang)
        normalizer, network = parseAndTrain(corpus, loadFolderPath)
        parse(corpus, network, normalizer)
        evaluate(corpus)


def identifyLinearKeras():
    for lang in langs:
        corpus = Corpus(lang)
        oracle.parse(corpus)
        normalizer = lkm.Normalizer(corpus)
        network = lkm.LinearKerasModel(len(normalizer.tokens) + len(normalizer.pos))
        lkm.train(network.model, corpus, normalizer)
        parse(corpus, network, normalizer)
        evaluate(corpus)


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


def parseAndTrain(corpus, loadFolderPath=''):
    if configuration["evaluation"]["load"]:
        normalizer = reports.loadNormalizer(loadFolderPath)
        logging.warn('Vocabulary size:{0}'.format(len(normalizer.vocabulary.indices)))
        network = init(corpus, normalizer=normalizer)
        network.model = reports.loadModel(loadFolderPath)
    else:
        if not configuration["evaluation"]["cv"]["active"]:
            reports.createReportFolder(corpus.langName)
        oracle.parse(corpus)
        normalizer, network = init(corpus)
        train(network.model, normalizer, corpus)
    return normalizer, network


def init(corpus, normalizer=None):
    # emb = configuration["model"]["embedding"]["active"]
    # mlp = configuration["model"]["topology"]["mlp"]["active"]
    # rnn = configuration["model"]["topology"]["rnn"]["active"]
    # feats = configuration["features"]["active"]

    if not normalizer:
        normalizer = Normalizer(corpus)
    network = Network(normalizer)
    return normalizer, network
    # if settings.USE_MODEL_CONF_TRANS:
    #     network = modelConfTrans.NetworkConfTrans(normalizer)
    #     return network

    # if mlp and feats and not emb and not rnn:
    #     if not normalizer:
    #         normalizer = modelMlpAuxSimple.Normalizer(corpus)
    #     network = modelMlpAuxSimple.Networ(normalizer)
    #     reports.createHeader('MLP (Features)')
    #     return normalizer, network

    # if settings.USE_MODEL_LSTM_IN_TWO_DIRS:
    #     network = modelLstmInTwoDirections.NetworkLstmInTwoDirections(normalizer)
    #     return network
    # if mlp and emb and not feats and not rnn:
    #     if not normalizer:
    #         normalizer = modelMlpHyperSimple.Normalizer(corpus)
    #     network = modelMlpHyperSimple.Network(normalizer)
    #     reports.createHeader('MLP (Words)')
    #     return normalizer, network
    # if mlp and emb and feats and not rnn:
    #     if not normalizer:
    #         normalizer = modelMlpSimple.Normalizer(corpus)
    #     network = modelMlpSimple.Network(normalizer)
    #     reports.createHeader('MLP (Features + Words)')
    #     return normalizer, network
    # if emb and mlp and feats and rnn:
    #     if not normalizer:
    #         normalizer = modelMlpLstm.Normalizer(corpus)
    #     network = modelMlpLstm.Network(normalizer)
    #     reports.createHeader('MLP LSTM (Features + Words)')
    #     return normalizer, network
    # if settings.USE_MODEL_MLP_LSTM_2:
    #     network = modelMLPLSTM2.NetworkMlpLstm2(normalizer)
    #     return network
    # raise ValueError('initNetword: No model Selected!')


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


def xp(train=False, cv=False, xpNum=1):
    evlaConf = configuration["evaluation"]
    evlaConf["cluster"] = True
    if train:
        ######################################
        #   Debug
        ######################################
        # evlaConf["debug"] = True
        # evlaConf["train"] = False
        # identify()
        ######################################
        #   Train
        ######################################
        evlaConf["debug"] = False
        evlaConf["train"] = True
        for i in range(xpNum):
            seed += 1
            numpy.random.seed(seed)
            random.seed(seed)
            identify()
        evlaConf["debug"] = True
        evlaConf["train"] = False
    if cv:
        ######################################
        #   CV Debug
        ######################################
        crossValidation(debug=True)
        ######################################
        #   CV
        ######################################
        for i in range(xpNum):
            global seed
            seed += 1
            numpy.random.seed(seed)
            random.seed(seed)
            crossValidation()
        ######################################
        #   Load
        ######################################
        # preTrainedPath= '/home/halsaied/nancy/NNIdenSys/NNIdenSys/Reports/FR-12/12-FR-modelWeigth.hdf5'
        # identify(load=configuration["evaluation"]["load"], loadFolderPath=loadFolderPath)


def xpGRU(stacked=False, gru=False, cv=False):
    rnnConf = configuration["model"]["topology"]["rnn"]
    rnnConf["active"] = True
    rnnConf["gru"] = True
    rnnConf["stacked"] = True
    title = '' if not stacked else 'Stacked '
    title += ' LSTM ' if not gru else ' GRU '
    reports.createHeader(title)
    if cv:
        xp(cv=True)
    else:
        xp(train=True)

    rnnConf["active"] = False
    rnnConf["gru"] = False
    rnnConf["stacked"] = False


def xpMLPAux(cv=False):
    configuration["model"]["embedding"]["active"] = False
    reports.createHeader('MLP Aux')
    if cv:
        xp(cv=True)
    else:
        xp(train=True)


def xpMLPTotal(train=False, cv=False, xpNum=5):
    configuration["evaluation"]["debug"] = False
    configuration["evaluation"]["train"] = True

    configuration["features"]["active"] = True
    configuration["model"]["topology"]["mlp"]["active"] = False
    configuration["model"]["embedding"]["active"] = False
    configuration["model"]["embedding"]["initialisation"] = False
    configuration["model"]["embedding"]["concatenation"] = False

    reports.createHeader('Features Only')
    # xp(train=train, cv=cv, xpNum=xpNum)

    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = True

    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 256
    reports.createHeader('Features Only + Dense_1(256)')
    # xp(train=train, cv=cv, xpNum=xpNum)

    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 512
    reports.createHeader('Features Only + Dense_1(512)')
    # xp(train=train, cv=cv, xpNum=xpNum)

    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = False
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 1024

    configuration["model"]["embedding"]["active"] = True
    configuration["model"]["embedding"]["pos"] = False
    reports.createHeader('Features + Tokens')
    # xp(train=train, cv=cv, xpNum=xpNum)

    configuration["model"]["embedding"]["tokenEmb"] = 100
    reports.createHeader('Features + Tokens (emb = 100)')
    # xp(train=train, cv=cv, xpNum=xpNum)
    configuration["model"]["embedding"]["tokenEmb"] = 200

    reports.createHeader('Features + Tokens  + Dense 128')
    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = True
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 128
    # xp(train=train, cv=cv, xpNum=xpNum)

    reports.createHeader('Features + Tokens  + Dense 256')
    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = True
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 256
    # xp(train=train, cv=cv, xpNum=xpNum)

    reports.createHeader('Features + Tokens  + Dense 128 + LSTM 256')
    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = True
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 128
    configuration["model"]["topology"]["rnn"]["active"] = True
    configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"] = 256
    # xp(train=train, cv=cv, xpNum=xpNum)
    configuration["model"]["topology"]["rnn"]["active"] = False
    configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"] = 512

    configuration["model"]["embedding"]["initialisation"] = True
    reports.createHeader('Features + Tokens + initialisation')
    # xp(train=train, cv=cv, xpNum=xpNum)
    configuration["model"]["embedding"]["initialisation"] = False

    configuration["model"]["embedding"]["pos"] = True
    reports.createHeader('Features + Tokens + POS')
    xp(train=train, cv=cv, xpNum=xpNum)

    configuration["model"]["embedding"]["posEmb"] = 50
    configuration["model"]["embedding"]["tokenEmb"] = 100
    reports.createHeader('Features + Tokens 100 + POS 50')
    xp(train=train, cv=cv, xpNum=xpNum)
    configuration["model"]["embedding"]["posEmb"] = 25
    configuration["model"]["embedding"]["tokenEmb"] = 200

    configuration["model"]["embedding"]["initialisation"] = True
    reports.createHeader('Features + Tokens + POS + initialisation')
    xp(train=train, cv=cv, xpNum=xpNum)

    configuration["model"]["embedding"]["initialisation"] = False
    reports.createHeader('Features + Tokens + POS + Dense 128')
    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = True
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 128
    xp(train=train, cv=cv, xpNum=xpNum)

    reports.createHeader('Features + Tokens + POS + Dense 256')
    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = True
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 256
    xp(train=train, cv=cv, xpNum=xpNum)
    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = False

    reports.createHeader('Features + Concatenation')
    configuration["model"]["embedding"]["initialisation"] = False
    configuration["model"]["embedding"]["concatenation"] = True
    xp(train=train, cv=cv, xpNum=xpNum)

    reports.createHeader('Features + Concatenation + Initialisation')
    configuration["model"]["embedding"]["initialisation"] = True
    xp(train=train, cv=cv, xpNum=xpNum)

    reports.createHeader('Features + Concatenation + Initialisation + Dense 128')
    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = True
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 128
    xp(train=train, cv=cv, xpNum=xpNum)

    reports.createHeader('Features + Concatenation + Initialisation + Dense 256')
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 256
    xp(train=train, cv=cv, xpNum=xpNum)

    reports.createHeader('Features + Concatenation + Initialisation + Dense 512')
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = 512
    xp(train=train, cv=cv, xpNum=xpNum)

    configuration["features"]["active"] = True
    configuration["model"]["embedding"]["active"] = True
    configuration["model"]["embedding"]["initialisation"] = False
    configuration["model"]["embedding"]["concatenation"] = True
    configuration["model"]["topology"]["rnn"]["active"] = True
    configuration["model"]["topology"]["rnn"]["gru"] = True
    reports.createHeader('Features + Concatenation + Dense 256 + gru 512')
    xp(train=train, cv=cv, xpNum=xpNum)

    reports.createHeader('Features + Concatenation + Dense 256 + gru 256')
    configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"] = 256
    xp(train=train, cv=cv, xpNum=xpNum)

    reports.createHeader('Features + Concatenation + Dense 256 + lstm 256')
    configuration["model"]["topology"]["rnn"]["gru"] = False
    xp(train=train, cv=cv, xpNum=xpNum)

    # reports.createHeader('Standard')
    # xp(cv=cv, train=train, xpNum=xpNum)
    #
    # reports.createHeader('Dense 2 not activated')
    # mlpConfig["dense2"]["active"] = False
    # xp(cv=cv, train=train, xpNum=xpNum)
    # mlpConfig["dense2"]["active"] = True
    #
    # reports.createHeader('Dense 3 activated')
    # mlpConfig["dense3"]["active"] = True
    # xp(cv=cv, train=train, xpNum=xpNum)
    # mlpConfig["dense3"]["active"] = False
    #
    # reports.createHeader('Tahn is activated')
    # mlpConfig["dense1"]["activation"] = "tanh"
    # mlpConfig["dense2"]["activation"] = "tanh"
    # xp(cv=cv, train=train, xpNum=xpNum)
    # mlpConfig["dense1"]["activation"] = "relu"
    # mlpConfig["dense2"]["activation"] = "relu"
    #
    # embConfig = configuration["model"]["embedding"]
    #
    # reports.createHeader('Without weight matrix')
    # embConfig["initialisation"] = False
    # xp(cv=cv, train=train, xpNum=xpNum)
    # embConfig["initialisation"] = True
    #
    # reports.createHeader('Maximized weight matrix')
    # embConfig["frequentTokens"] = False
    # xp(cv=cv, train=train, xpNum=xpNum)
    # embConfig["frequentTokens"] = True
    #
    # reports.createHeader('Use Lemma')
    # embConfig["lemma"] = True
    # xp(cv=cv, train=train, xpNum=xpNum)
    # embConfig["lemma"] = False
    #
    # reports.createHeader('No POS emb')
    # embConfig["pos"] = False
    # xp(cv=cv, train=train, xpNum=xpNum)
    # embConfig["pos"] = True
    #
    # trainConfig = configuration["model"]["train"]
    #
    # reports.createHeader('Batch size = 64')
    # trainConfig["batchSize"] = 64
    # xp(cv=cv, train=train, xpNum=xpNum)
    # trainConfig["batchSize"] = 128
    #
    # reports.createHeader('Batch size = 256')
    # trainConfig["batchSize"] = 256
    # xp(cv=cv, train=train, xpNum=xpNum)
    # trainConfig["batchSize"] = 128
    #
    # reports.createHeader('No early stop')
    # trainConfig["earlyStop"] = False
    # xp(cv=cv, train=train, xpNum=xpNum)
    # trainConfig["earlyStop"] = True


# def xpConfTrans(stacked=False, gru=False, cv=False):
#     settings.USE_MODEL_CONF_TRANS = True
#     configuration["model"]["topology"]["rnn"]["stacked"] = stacked
#     configuration["model"]["topology"]["rnn"]["gru"] = gru
#     title = 'Conf <=> Trans: '
#     title += '' if not stacked else 'Stacked '
#     title += ' LSTM ' if not gru else ' GRU '
#     reports.createHeader(title)
#     if cv:
#         xp(cv=True)
#     else:
#         xp(train=True)
#
#     settings.USE_MODEL_CONF_TRANS = False
#     configuration["model"]["topology"]["rnn"]["stacked"] = False
#     configuration["model"]["topology"]["rnn"]["gru"] = False


def linearModelImpact():
    evalConfig = configuration["evaluation"]
    evalConfig["debug"] = False
    evalConfig["train"] = True

    featConf = configuration["features"]

    reports.createHeader('Standard settings:  A B C E I J K L')
    identifyV2()

    reports.createHeader('Without syntax: A C E I J K L')
    featConf["syntax"]["active"] = False
    identifyV2()
    featConf["syntax"]["active"] = True

    reports.createHeader('Without BiGram: A B E I J K L')
    featConf["bigram"]["s0b2"] = False
    identifyV2()
    featConf["bigram"]["s0b2"] = True

    reports.createHeader('Without S0B2Bigram: A B C I J K L')
    featConf["bigram"]["s0b2"] = False
    identifyV2()
    featConf["bigram"]["s0b2"] = True

    reports.createHeader('Without S0B0Distance: A B C E J K L')
    featConf["distance"]["s0b0"] = False
    identifyV2()
    featConf["distance"]["s0b0"] = True

    reports.createHeader('Without S0S1Distance: A B C E I K L')
    featConf["distance"]["s0s1"] = False
    identifyV2()
    featConf["distance"]["s0s1"] = True

    reports.createHeader('without B1:  A B C E I J L')
    featConf["unigram"]["b1"] = False
    identifyV2()
    featConf["unigram"]["b1"] = True

    reports.createHeader('without lexicon:  A B C E I J K')
    featConf["dictionary"]["active"] = False
    identifyV2()
    featConf["dictionary"]["active"] = False

    featConf["syntax"]["active"] = False
    featConf["bigram"]["active"] = False

    featConf["dictionary"]["active"] = False
    featConf["unigram"]["b1"] = False
    featConf["bigram"]["s0b2"] = False
    featConf["distance"]["s0s1"] = False
    featConf["distance"]["s0b0"] = False
    reports.createHeader('Unigram only:  A ')
    identifyV2()

    featConf["unigram"]["pos"] = False
    featConf["unigram"]["lemma"] = False
    reports.createHeader('token  only')
    identifyV2()

    reports.createHeader('lemma + token ')
    featConf["unigram"]["lemma"] = True
    identifyV2()

    reports.createHeader('Pos + token')
    featConf["unigram"]["lemma"] = False
    featConf["unigram"]["pos"] = True
    identifyV2()


global seed
seed = 0
reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.WARNING)
numpy.random.seed(seed)
random.seed(seed)

# Standard configuration:
# Evaluation: Debug, Cluster
# Embedding: Active POS Concatenation frequentTokens
# TOPOLOGY: MLP NO DEENSE
# Features: ALL except(suffix + mwt)

identifyLinearKeras()
# xpMLPTotal(train=True, xpNum=5)


# configuration["evaluation"]["debug"] = False
# configuration["evaluation"]["train"] = True
#
# configuration["features"]["active"] = True
# configuration["model"]["embedding"]["active"] = True
# configuration["model"]["embedding"]["initialisation"] = True
# configuration["model"]["embedding"]["concatenation"] = True
# configuration["model"]["topology"]["rnn"]["active"] = True
# configuration["model"]["topology"]["rnn"]["gru"] = True
# configuration["model"]["topology"]["rnn"]["rnn1"]["unitNumber"] = 256
#
# identify()
