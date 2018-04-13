import random
import sys

import numpy

import reports
from config import *
from deepExpirements import desactivateMainConf
from identification import identifyAttached, crossValidation


def xp(debug=True, train=False, cv=False, xpNum=10, title=''):
    evlaConf = configuration["evaluation"]
    evlaConf["cluster"] = True
    global seed
    seed = 0
    if debug and not train:
        ######################################
        #   Debug
        ######################################
        evlaConf["debug"] = True
        evlaConf["train"] = False
        sys.stdout.write('Debug enabled\n')
        if title:
            reports.createHeader(title)
        identifyAttached()
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
        sys.stdout.write('#' * 50 + '\n')
        sys.stdout.write('#' * 50 + '\n')
        sys.stdout.write('#' * 50 + '\n')
        sys.stdout.write("CV is not available yet for this model!\n")
        sys.stdout.write('#' * 50 + '\n')
        sys.stdout.write('#' * 50 + '\n')
        sys.stdout.write('#' * 50 + '\n')
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
            mlpConf = configuration["model"]["mlp"]
            mlpConf["active"] = True


def exploreDenseImpact(denseUniNumDomain, tokenEmb, posEmb, train=False, cv=False, xpNum=10, title=''):
    desactivateMainConf()
    configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
    configuration["model"]["embedding"]["posEmb"] = posEmb
    mlpConf = configuration["model"]["mlp"]
    mlpConf["dense1"]["active"] = True
    for denseUniNum in denseUniNumDomain:
        title = '{0} Dense {1}'.format(title, denseUniNum)
        mlpConf["dense1"]["unitNumber"] = denseUniNum
        xp(train=train, cv=cv, xpNum=xpNum, title=title)


def tableEmb(useFeatures=False, train=False, cv=False, xpNum=10):
    tokenDmain = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    posDomain = [8, 16, 24, 32, 40, 48, 56, 64, 72]
    title = ''
    desactivateMainConf()
    if useFeatures:
        configuration["features"]["active"] = True
        title = 'Features '
    setEmbConf(usePos=False, init=False)
    exploreEmbImpact(tokenDmain, train=train, cv=cv, xpNum=xpNum, title=title)
    exploreEmbImpact(tokenDmain, useLemma=True, train=train, cv=cv, xpNum=xpNum, title=title)
    exploreEmbImpact(tokenDmain, posDomain, usePos=True, train=train, cv=cv, xpNum=xpNum, title=title)
    exploreEmbImpact(tokenDmain, posDomain, usePos=True, useLemma=True, train=train, cv=cv, xpNum=xpNum, title=title)


def exploreEmbImpact(tokenEmbs, posEmbs=None, useLemma=False, usePos=False, train=False, cv=False, xpNum=10, title=''):
    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["usePos"] = usePos
    embConf["lemma"] = useLemma
    for tokenEmb in tokenEmbs:
        if usePos:
            for posEmb in posEmbs:
                embConf["posEmb"] = posEmb
                embConf["tokenEmb"] = tokenEmb
                newtitle = '{0}({1}) POS({2}) {3}'.format('Lemma' if useLemma else 'Token', tokenEmb, posEmb, title)
                xp(train=train, cv=cv, xpNum=xpNum, title=newtitle)
        else:
            embConf["tokenEmb"] = tokenEmb
            newtitle = '{0}({1}) {2}'.format('Lemma' if useLemma else 'Token', tokenEmb, title)
            xp(train=train, cv=cv, xpNum=xpNum, title=newtitle)

    embConf["usePos"] = False
    embConf["lemma"] = False


if __name__ == '__main__':
    configuration["model"]["padding"] = False

    tokenDmain = [125, 150, 175, 200, 225, 250]
    posDomain = [24, 32, 40, 48, 56, 64]
    desactivateMainConf()
    setEmbConf(usePos=True, init=False)
    exploreEmbImpact(tokenDmain, posDomain, useLemma=False, usePos=True, train=True)

    # tableEmb(train=True)
    # tableEmb(train=True,useFeatures=True)

    # 1. explore token pos impact
    # exploreTokenPOSImpact([25, 50, 75, 100, 125, 150, 175, 200], [8, 16, 24, 32, 40, 48, 56], train=True)

    ## 2. Add features as input:
    # configuration["features"]["active"] = True
    # resetFRStandardFeatures()
    # tokenEmb, posEmb = 175, 56
    # configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
    # configuration["model"]["embedding"]["posEmb"] = posEmb
    # # xp(train=True, title='Token {0} POS {1} Features'.format(tokenEmb, posEmb))
    # configuration["features"]["active"] = False
    #
    # ## 3. Add dense to get bigram effect
    # configuration["model"]["embedding"]["active"] = True
    # # exploreDenseImpact([32,64,96,128,160,256, 384, 512,768, 1024], tokenEmb=175, posEmb=56, train=True)
    #
    # ## 4. Standard Model: Token POS Features  Dense
    # configuration["features"]["active"] = True
    # tokenEmb, posEmb, denseUnitNum = 175, 56, 768
    # configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
    # configuration["model"]["embedding"]["posEmb"] = posEmb
    # configuration["model"]["mlp"]["dense1"]["active"] = True
    # configuration["model"]["mlp"]["dense1"]["unitNumber"] = denseUnitNum
    # title = 'Token {0} POS {1} Dense {2} Features'.format(tokenEmb, posEmb, denseUnitNum)
    # xp(train=True, title=title)
