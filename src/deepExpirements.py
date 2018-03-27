import random
import sys

import numpy

from corpus import *
from identification import identify, crossValidation

allLangs = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT', 'LT', 'MT', 'PL', 'PT', 'RO', 'SL', 'SV', 'TR']


def xp(train=False, cv=False, xpNum=1, title=''):
    evlaConf = configuration["evaluation"]
    evlaConf["cluster"] = True
    global seed
    seed = 0
    if train:
        ######################################
        #   Debug
        ######################################
        evlaConf["debug"] = True
        evlaConf["train"] = False
        if title:
            reports.createHeader(title)
        identify()
        ######################################
        #   Train
        ######################################
        evlaConf["debug"] = False
        evlaConf["train"] = True
        for i in range(xpNum):
            numpy.random.seed(seed)
            random.seed(seed)
            if title:
                reports.createHeader(title)
            identify()
            seed += 1
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
            seed += 1
            numpy.random.seed(seed)
            random.seed(seed)
            if title:
                reports.createHeader(title)
            crossValidation()
            ######################################
            #   Load
            ######################################
            # preTrainedPath= '/home/halsaied/nancy/NNIdenSys/NNIdenSys/Reports/FR-12/12-FR-modelWeigth.hdf5'
            # identify(load=configuration["evaluation"]["load"], loadFolderPath=loadFolderPath)


def exploreTokenPosEmbImpact(domain, train=False, cv=False, xpNum=5, usePos=False):
    desactivateMainConf()
    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["usePos"] = usePos
    posTitle = '+ POS' if usePos else ''
    for embDim in domain:
        if usePos:
            configuration["model"]["embedding"]["posEmb"] = embDim
            title = 'Tokens ( {0} emb = {1})'.format(posTitle, embDim)
        else:
            configuration["model"]["embedding"]["tokenEmb"] = embDim
            title = 'Tokens {0} (emb = {1})'.format(posTitle, embDim)
        xp(train=train, cv=cv, xpNum=xpNum, title=title)


def exploreAuxFeatureImpact(denseDomain, train=False, cv=False, xpNum=5):
    desactivateMainConf()
    configuration["features"]["active"] = True
    xp(train=train, cv=cv, xpNum=xpNum, title='Features')
    exploreDenseUnitNum(denseDomain, train=train, cv=cv, xpNum=xpNum, title='Features')
    configuration["features"]["active"] = False


def exploreDenseUnitNum(denseDomain, layer="dense1", train=False, cv=False, xpNum=5, title=''):
    configuration["model"]["topology"]["mlp"]["active"] = True
    dense1Conf = configuration["model"]["topology"]["mlp"][layer]
    dense1Conf["active"] = True
    for unitNum in denseDomain:
        dense1Conf["unitNumber"] = unitNum
        xp(train=train, cv=cv, xpNum=xpNum, title='{0} + Dense {1}'.format(title, unitNum))
    dense1Conf["active"] = False


def exploreRnnUnitNum(rnnDomain, train=False, cv=False, xpNum=5, title=''):
    rnnConf = configuration["model"]["topology"]["rnn"]["rnn1"]
    rnnConf["active"] = True
    for unitNum in rnnDomain:
        rnnConf["unitNumber"] = unitNum
        xp(train=train, cv=cv, xpNum=xpNum, title='{0} + RNN {1}'.format(title, unitNum))
    rnnConf["active"] = False


def xpGRU(stacked=False, gru=False, cv=False):
    rnnConf = configuration["model"]["topology"]["rnn"]
    rnnConf["active"] = True
    rnnConf["gru"] = True
    rnnConf["stacked"] = True
    title = '' if not stacked else 'Stacked '
    title += ' LSTM ' if not gru else ' GRU '
    xp(cv=cv, train=not cv, title=title)
    rnnConf["active"] = False
    rnnConf["gru"] = False
    rnnConf["stacked"] = False


def exploreMLPDepth(train=False, cv=False, xpNum=5, title=''):
    pass
    # TODO
    # mlpConfig = configuration["model"]["topology"]["mlp"]
    # reports.createHeader('Dense 2 not activated')
    # mlpConfig["dense2"]["active"] = False
    # xp(cv=cv, train=train, xpNum=xpNum)
    # mlpConfig["dense2"]["active"] = True
    #
    # reports.createHeader('Dense 3 activated')
    # mlpConfig["dense3"]["active"] = True
    # xp(cv=cv, train=train, xpNum=xpNum)
    # mlpConfig["dense3"]["active"] = False


def exploreActivationFunImpact(train=False, cv=False, xpNum=5, title=''):
    pass
    # TODO
    # mlpConfig = configuration["model"]["topology"]["mlp"]
    # reports.createHeader('Tahn is activated')
    # mlpConfig["dense1"]["activation"] = "tanh"
    # mlpConfig["dense2"]["activation"] = "tanh"
    # xp(cv=cv, train=train, xpNum=xpNum)
    # mlpConfig["dense1"]["activation"] = "relu"
    # mlpConfig["dense2"]["activation"] = "relu"


def exploreWeighMatrixImpact(train=False, cv=False, xpNum=5):
    desactivateMainConf()
    configuration["model"]["embedding"]["active"] = True
    configuration["model"]["embedding"]["usePos"] = True

    configuration["model"]["embedding"]["tokenEmb"] = 200
    configuration["model"]["embedding"]["posEmb"] = 50
    # without weight matrix
    xp(train=train, cv=cv, xpNum=xpNum, title='Token + POS + No initialisation')
    configuration["model"]["embedding"]["initialisation"]["active"] = True
    # with weight matrix
    xp(train=train, cv=cv, xpNum=xpNum, title='Token + POS + initialisation')

    # with maximized weight matrix
    configuration["model"]["embedding"]["frequentTokens"] = False
    configuration["model"]["embedding"]["initialisation"]["active"] = False
    xp(train=train, cv=cv, xpNum=xpNum, title='Token POS maximized + no inisialisation')
    configuration["model"]["embedding"]["initialisation"]["active"] = True
    xp(train=train, cv=cv, xpNum=xpNum, title='Token POS maxilized + initilaisation')
    configuration["model"]["embedding"]["frequentTokens"] = True

    # with lemma-based weight matrix without initialisation
    configuration["model"]["embedding"]["lemma"] = True
    configuration["model"]["embedding"]["initialisation"]["active"] = False
    xp(train=train, cv=cv, xpNum=xpNum, title='Lemma + POS + No initialisation')
    # with lemma-based weight matrix without initialisation
    configuration["model"]["embedding"]["initialisation"]["active"] = True
    xp(train=train, cv=cv, xpNum=xpNum, title='Lemma + POS + initialisation')
    configuration["model"]["embedding"]["lemma"] = False


def exploreTrainParamsImpact(train=False, cv=False, xpNum=5, title=''):
    trainConfig = configuration["model"]["train"]
    exploreBatchSizeImpact([32, 64, 128, 256, 512], train=train, cv=cv, xpNum=xpNum, title=title)

    trainConfig["earlyStop"] = False
    xp(train=train, cv=cv, xpNum=xpNum, title='{0} + No Early Stop'.format(title))
    trainConfig["earlyStop"] = True


def exploreBatchSizeImpact(domain, train=False, cv=False, xpNum=5, title=''):
    for item in domain:
        configuration["model"]["train"]["batchSize"] = item
        xp(train=train, cv=cv, xpNum=xpNum, title='{0} + BatchSize {1}'.format(title, item))
    configuration["model"]["train"]["batchSize"] = 128


def desactivateMainConf():
    configuration["features"]["active"] = False

    configuration["model"]["topology"]["mlp"]["active"] = False
    configuration["model"]["topology"]["rnn"]["active"] = False

    configuration["model"]["embedding"]["active"] = False
    configuration["model"]["embedding"]["initialisation"]["active"] = False
    configuration["model"]["embedding"]["concatenation"] = False
    configuration["model"]["embedding"]["usePos"] = False


def tokenPOSEmbImpact():
    exploreTokenPosEmbImpact([25, 50, 100, 150, 200, 250, 300], train=True, xpNum=10, usePos=False)
    configuration["model"]["embedding"]["tokenEmb"] = 50
    exploreTokenPosEmbImpact([8, 16, 24, 32, 40, 48, 56], train=True, xpNum=10, usePos=True)


def exploreDense1Impact():
    configuration["model"]["embedding"]["active"] = True
    configuration["model"]["embedding"]["usePos"] = True
    configuration["model"]["embedding"]["tokenEmb"] = 200
    configuration["model"]["embedding"]["posEmb"] = 50
    exploreDenseUnitNum([32, 64, 128, 256, 512, 1024], train=True, title='')


def exploreBinaryPOSEmb(domain, train=False, cv=False, xpNum=5, title=''):
    desactivateMainConf()
    configuration["model"]["embedding"]["active"] = True
    configuration["model"]["embedding"]["usePos"] = True
    configuration["model"]["embedding"]["initialisation"]["active"] = True
    configuration["model"]["embedding"]["initialisation"]["oneHotPos"] = True
    configuration["model"]["embedding"]["initialisation"]["token"] = False

    for embDim in domain:
        configuration["model"]["embedding"]["tokenEmb"] = embDim
        xp(train=train, cv=cv, xpNum=xpNum, title='{0} Token {1}+ Binary POS Emb '.format(title, embDim))


reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.WARNING)

# tokenPOSEmbImpact()
# exploreWeighMatrixImpact(train=True)
# exploreDense1Impact()

exploreBinaryPOSEmb([25, 50, 100, 150, 200], train=True)