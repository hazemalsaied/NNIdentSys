import random
import sys

import numpy

import reports
from config import *
from deepExpirements import desactivateMainConf
from identification import identifyAttached, crossValidation


def xp(train=False, cv=False, xpNum=10, title='', langs=['FR'], initSeed=0):
    evlaConf = configuration["evaluation"]
    evlaConf["cluster"] = True
    global seed
    seed = 0
    if not cv and not train:
        ######################################
        #   Debug
        ######################################
        evlaConf["debug"] = True
        evlaConf["train"] = False
        for lang in langs:
            sys.stdout.write('Debug enabled\n')
            title1 = lang +': ' +  title
            reports.createHeader(title1)
            identifyAttached(lang)
    if train:
        ######################################
        #   Train
        ######################################
        evlaConf["debug"] = False
        evlaConf["train"] = True
        for lang in langs:
            seed = initSeed
            title1 = lang + ': ' + title
            for i in range(xpNum):
                numpy.random.seed(seed)
                random.seed(seed)
                reports.createHeader(title1)
                identifyAttached(lang)
                seed += 1
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
        crossValidation(langs=langs, debug=True)
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


def exploreTokenPOSImpact(tokenDomain, posDomain, train=False, cv=False, xpNum=10, langs=['FR']):
    desactivateMainConf()
    configuration["model"]["embedding"]["active"] = True
    for tokenEmb in tokenDomain:
        for posEmb in posDomain:
            configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
            configuration["model"]["embedding"]["posEmb"] = posEmb
            title = 'Token {0}, POS {1}'.format(tokenEmb, posEmb)
            xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=title)
            mlpConf = configuration["model"]["mlp"]
            mlpConf["active"] = True


def exploreDenseImpact(denseUniNumDomain, tokenEmb, posEmb, train=False, cv=False, xpNum=10, title='', langs=['FR']):
    desactivateMainConf()
    configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
    configuration["model"]["embedding"]["posEmb"] = posEmb
    mlpConf = configuration["model"]["mlp"]
    mlpConf["dense1"]["active"] = True
    for denseUniNum in denseUniNumDomain:
        title = '{0} Dense {1}'.format(title, denseUniNum)
        mlpConf["dense1"]["unitNumber"] = denseUniNum
        xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=title)


def tableEmb(useFeatures=False, train=False, cv=False, xpNum=10, langs=['FR']):
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


def exploreEmbImpact(tokenEmbs, posEmbs=None, useLemma=False, usePos=False, train=False, cv=False, xpNum=10, title='',
                     langs=['FR']):
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
                xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=newtitle)
        else:
            embConf["tokenEmb"] = tokenEmb
            newtitle = '{0}({1}) {2}'.format('Lemma' if useLemma else 'Token', tokenEmb, title)
            xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=newtitle)

    embConf["usePos"] = False
    embConf["lemma"] = False


def denseImpact(useLemma=False, train=True, langs=['FR']):
    embConf = configuration["model"]["embedding"]
    desactivateMainConf()
    tokenEmbDic = {}
    tokenEmbDic[125] = [56, 64]
    tokenEmbDic[150] = [64]
    tokenEmbDic[175] = [64]
    denseDomain = [64, 96, 128, 192, 256, 384, 512, 1024]
    setEmbConf(usePos=True, init=False)
    embConf["lemma"] = useLemma
    for tokenEmb in tokenEmbDic.keys():
        embConf["tokenEmb"] = tokenEmb
        for posEmb in tokenEmbDic[tokenEmb]:
            embConf["posEmb"] = posEmb
            for denseUnitNum in denseDomain:
                setDense1Conf(unitNumber=denseUnitNum)
                res = 'Lemma' if useLemma else 'Token'
                xp(langs=langs, train=train,
                   title=res + ' {0} POS {1} Dense {2}'.format(tokenEmb, posEmb, denseUnitNum))


def dense2Impact(useLemma=False, train=True, langs=['FR']):
    embConf = configuration["model"]["embedding"]
    desactivateMainConf()
    if useLemma:
        tokenEmb = 125
        posEmb = 56
        dense1UniNum = 256
    else:
        tokenEmb = 175
        posEmb = 64
        dense1UniNum = 192
    dense2Domain = [96, 128, 192, 256, 384]
    setEmbConf(usePos=True, init=False)
    embConf["lemma"] = useLemma
    embConf["tokenEmb"] = tokenEmb
    embConf["posEmb"] = posEmb
    setDense1Conf(unitNumber=dense1UniNum)
    configuration["model"]["mlp"]["dense2"]["active"] = True
    res = 'Lemma' if useLemma else 'Token'
    for denseUnitNum in dense2Domain:
        configuration["model"]["mlp"]["dense2"]["unitNumber"] = denseUnitNum
        xp(langs=langs, train=train, title=res + ' {0} POS {1} Dense1 {2} Dense2 {3}'.
           format(tokenEmb, posEmb, dense1UniNum, denseUnitNum))


def exploreInitParams(useLemma=False, train=True, langs=['FR']):
    embConf = configuration["model"]["embedding"]
    desactivateMainConf()
    tokenEmbs = [200, 250]
    posEmbs = [64, 80, 96, 128]
    dense1UniNums = [256, 384, 512]
    embConf["lemma"] = useLemma
    setEmbConf(usePos=True, init=False)
    for tokenEmb in tokenEmbs:
        embConf["tokenEmb"] = tokenEmb
        for posEmb in posEmbs:
            embConf["posEmb"] = posEmb
            for dense1UnitNum in dense1UniNums:
                setDense1Conf(unitNumber=dense1UnitNum)
                res = 'Lemma' if useLemma else 'Token'
                xp(langs=langs, train=train, title=res + ' {0} POS {1} Dense1 {2}'.
                   format(tokenEmb, posEmb, dense1UnitNum))


def initImpact(tokenEmb, posEmb, DenseUnitNum, initType="frWac200", useLemma=False, train=True,xpNum=10, langs=['FR']):
    embConf = configuration["model"]["embedding"]
    setEmbConf(tokenEmb=tokenEmb, posEmb=posEmb, usePos=True, init=True)
    embConf["lemma"] = useLemma
    embConf["initialisation"]["type"] = initType  # "frWac200"  # "dataFR.profiles.min.250"
    setDense1Conf(unitNumber=DenseUnitNum)
    maximisation = '' if configuration["model"]["embedding"]["frequentTokens"] else 'Maximised'
    xp(langs=langs, train=train, xpNum=xpNum, title='{0} {1} POS {2} Dense1 {3} init {4} {5}'.
       format('Lemma' if useLemma else 'Token', tokenEmb, posEmb, DenseUnitNum, initType, maximisation))


def initImpactTotal(train=False, langs=['FR']):
    desactivateMainConf()
    initImpact(200, 96, 384, initType="frWac200", train=train, langs=langs)
    initImpact(200, 96, 384, initType="frWac200", useLemma=True, train=train, langs=langs)
    configuration["model"]["embedding"]["frequentTokens"] = False
    initImpact(200, 96, 384, initType="frWac200", train=train, langs=langs)
    initImpact(200, 96, 384, initType="frWac200", useLemma=True, train=train, langs=langs)

    configuration["model"]["embedding"]["frequentTokens"] = True

    initImpact(250, 128, 384, initType="dataFR.profiles.min.250", train=train, langs=langs)
    initImpact(250, 128, 512, initType="dataFR.profiles.min.250", useLemma=True, train=train, langs=langs)
    configuration["model"]["embedding"]["frequentTokens"] = False
    initImpact(250, 128, 384, initType="dataFR.profiles.min.250", train=train, langs=langs)
    initImpact(250, 128, 512, initType="dataFR.profiles.min.250", useLemma=True, train=train, langs=langs)


def initImpactTotal2(train=False, langs=['FR'], xpNum=20):
    desactivateMainConf()
    initImpact(200, 56, 32, initType="frWac200", useLemma=True, train=train, langs=langs, xpNum=xpNum)
    initImpact(200, 64, 64, initType="frWac200", useLemma=True, train=train, langs=langs, xpNum=xpNum)

    initImpact(250, 56, 32, initType="dataFR.profiles.min.250", useLemma=True, train=train, langs=langs, xpNum=xpNum)
    initImpact(250, 64, 64, initType="dataFR.profiles.min.250", useLemma=True, train=train, langs=langs, xpNum=xpNum)


def exploreFRMax(train=False, initSeed=0):
    desactivateMainConf()
    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["lemma"] = True
    embConf["usePos"] = True
    embConf["tokenEmb"] = 125
    embConf["posEmb"] = 56
    dense1 = configuration["model"]["mlp"]["dense1"]
    dense1["active"] = True
    dense1["unitNumber"] = 256
    xp(langs=['FR'], train=train, xpNum=30, initSeed=initSeed)

def exploreAllLangs(shuffle=False):
    configuration["model"]["padding"] = False
    desactivateMainConf()
    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["lemma"] = True
    embConf["usePos"] = True
    embConf["tokenEmb"] = 250
    embConf["posEmb"] = 96
    dense1 = configuration["model"]["mlp"]["dense1"]
    dense1["active"] = True
    dense1["unitNumber"] = 512
    configuration["evaluation"]["shuffleTrain"] = shuffle
    xp(langs=['RO', 'CS', 'PT', 'TR', 'IT', 'LT', 'PL'], train=True)


def exploreRnn(train=True,useDense=True):
    desactivateMainConf()
    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["lemma"] = True
    embConf["usePos"] = True
    embConf["tokenEmb"] = 125
    embConf["posEmb"] = 56
    rnn1 = configuration["model"]["rnn"]
    rnn1['active'] = True
    rnn1['rnn1']['uitNumber'] = 256
    rnn1['rnn1']['posUitNumber'] = 128
    dense1 = configuration["model"]["mlp"]["dense1"]
    dense1["active"] = useDense
    dense1["unitNumber"] = 128
    xp(langs=['FR'], train=train)


if __name__ == '__main__':
    configuration["model"]["padding"] = False

    configuration["evaluation"]["shuffleTrain"] = True
    exploreRnn(True,False)