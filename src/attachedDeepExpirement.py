import logging
import random

import numpy

import reports
from config import configuration
from deepExpirements import desactivateMainConf
from identification import identifyAttached, crossValidation
from linearExpirements import resetFRStandardFeatures

def xp(train=False, cv=False, xpNum=10, title=''):
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
            identifyAttached()
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
            identifyAttached()
            seed += 1
        evlaConf["debug"] = True
        evlaConf["train"] = False
    if cv:
        logging.warn('#' * 50)
        logging.warn('#' * 50)
        logging.warn('#' * 50)
        logging.warn("CV is not available yet for this model!")
        logging.warn('#' * 50)
        logging.warn('#' * 50)
        logging.warn('#' * 50)
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
            # crossValidation()
            ######################################
            #   Load
            ######################################
            # preTrainedPath= '/home/halsaied/nancy/NNIdenSys/NNIdenSys/Reports/FR-12/12-FR-modelWeigth.hdf5'
            # identify(load=configuration["evaluation"]["load"], loadFolderPath=loadFolderPath)


def exploreTokenPOSImpact(tokenDomain, posDomain, train=False, cv=False, xpNum=10):
    desactivateMainConf()
    configuration["model"]["embedding"]["active"] = True
    for tokenEmb in tokenDomain:
        for posEmb in posDomain:
            configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
            configuration["model"]["embedding"]["posEmb"] = posEmb
            title = 'Token {0}, POS {1}'.format(tokenEmb, posEmb)
            xp(train=train, cv=cv, xpNum=xpNum, title=title)
            mlpConf = configuration["model"]["topology"]["mlp"]
            mlpConf["active"] = True
            mlpConf["dense1"]["active"] = True


def exploreDenseImpact(denseUniNumDomain, tokenEmb, posEmb, train=False, cv=False, xpNum=10, title=''):
    desactivateMainConf()
    configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
    configuration["model"]["embedding"]["posEmb"] = posEmb
    mlpConf = configuration["model"]["topology"]["mlp"]
    mlpConf["active"] = True
    mlpConf["dense1"]["active"] = True
    for denseUniNum in denseUniNumDomain:
        title = '{0} Dense {1}'.format(title, denseUniNum)
        mlpConf["dense1"]["unitNumber"] = denseUniNum
        xp(train=train, cv=cv, xpNum=xpNum, title=title)


if __name__ == '__main__':
    configuration["model"]["padding"] = False
    # 1. explore token pos impact
    #exploreTokenPOSImpact([25, 50, 75, 100, 125, 150, 175, 200], [8, 16, 24, 32, 40, 48, 56], train=True)

    ## 2. Add features as input:
    configuration["features"]["active"] = True
    resetFRStandardFeatures()
    tokenEmb, posEmb = 175, 56
    configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
    configuration["model"]["embedding"]["posEmb"] = posEmb
    #xp(train=True, title='Token {0} POS {1} Features'.format(tokenEmb, posEmb))
    configuration["features"]["active"] = False

    ## 3. Add dense to get bigram effect
    configuration["model"]["embedding"]["active"] = True
    #exploreDenseImpact([32,64,96,128,160,256, 384, 512,768, 1024], tokenEmb=175, posEmb=56, train=True)

    ## 4. Standard Model: Token POS Features  Dense
    configuration["features"]["active"] = True
    tokenEmb, posEmb, denseUnitNum = 175, 56, 768
    configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
    configuration["model"]["embedding"]["posEmb"] = posEmb
    configuration["model"]["topology"]["mlp"]["active"] = True
    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = True
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = denseUnitNum
    title = 'Token {0} POS {1} Dense {2} Features'.format(tokenEmb, posEmb, denseUnitNum)
    xp(train=True, title=title)
