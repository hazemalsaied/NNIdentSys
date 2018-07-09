import sys

from config import *
from identification import xp

langs = ['FR']
xpNum = 5
train = True
cv = False

allSharedtask1Lang = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT',
                      'LT', 'MT', 'PL', 'PT', 'RO', 'SV', 'SL', 'TR']

allSharedtask2Lang = ['BG', 'DE', 'EL', 'EN', 'ES', 'EU', 'FA', 'FR', 'HE', 'HI',
                      'HR', 'HU', 'IT', 'LT', 'PL', 'PT', 'RO', 'SL', 'TR']


def exploreTokenPOSImpact(tokenDomain, posDomain):
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


def exploreDenseImpact(denseUniNumDomain, tokenEmb, posEmb, title=''):
    desactivateMainConf()
    configuration["model"]["embedding"]["tokenEmb"] = tokenEmb
    configuration["model"]["embedding"]["posEmb"] = posEmb
    mlpConf = configuration["model"]["mlp"]
    mlpConf["dense1"]["active"] = True
    for denseUniNum in denseUniNumDomain:
        title = '{0} Dense {1}'.format(title, denseUniNum)
        mlpConf["dense1"]["unitNumber"] = denseUniNum
        xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=title)


def tableEmb(useFeatures=False):
    tokenDmain = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    posDomain = [8, 16, 24, 32, 40, 48, 56, 64, 72]
    title = ''
    desactivateMainConf()
    if useFeatures:
        configuration["features"]["active"] = True
        title = 'Features '
    setEmbConf(usePos=False, init=False)
    exploreEmbImpact(tokenDmain, title=title)
    exploreEmbImpact(tokenDmain, useLemma=True, title=title)
    exploreEmbImpact(tokenDmain, posDomain, usePos=True, title=title)
    exploreEmbImpact(tokenDmain, posDomain, usePos=True, useLemma=True, title=title)


def exploreEmbImpact(tokenEmbs, posEmbs=None, denseDomain=None, useLemma=False, usePos=False, title=''):
    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["usePos"] = usePos
    embConf["lemma"] = useLemma
    for tokenEmb in tokenEmbs:
        if usePos:
            for posEmb in posEmbs:
                embConf["posEmb"] = posEmb
                embConf["tokenEmb"] = tokenEmb
                if denseDomain:
                    for denseUnitNum in denseDomain:
                        newtitle = '{0}({1}) POS({2}) Dense {3} {4}'.format('Lemma' if useLemma else 'Token', tokenEmb,
                                                                            posEmb, denseUnitNum, title)
                        setDense1Conf(unitNumber=denseUnitNum)
                        xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=newtitle)
                else:
                    newtitle = '{0}({1}) POS({2}) Dense {3}'.format('Lemma' if useLemma else 'Token', tokenEmb,
                                                                    posEmb, title)
                    xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=newtitle)
        else:
            embConf["tokenEmb"] = tokenEmb
            newtitle = '{0}({1}) {2}'.format('Lemma' if useLemma else 'Token', tokenEmb, title)
            xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=newtitle)

    embConf["usePos"] = False
    embConf["lemma"] = False


def denseImpact(useLemma=False):
    embConf = configuration["model"]["embedding"]
    desactivateMainConf()
    tokenEmbDic = dict()
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


def dense2Impact(useLemma=False):
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


def exploreInitParams(useLemma=False):
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


def initImpact(tokenEmb, posEmb, DenseUnitNum, initType="frWac200", useLemma=False):
    embConf = configuration["model"]["embedding"]
    setEmbConf(tokenEmb=tokenEmb, posEmb=posEmb, usePos=True, init=True)
    embConf["lemma"] = useLemma
    embConf["initialisation"]["type"] = initType  # "frWac200"  # "dataFR.profiles.min.250"
    setDense1Conf(unitNumber=DenseUnitNum)
    maximisation = '' if configuration["model"]["embedding"]["frequentTokens"] else 'Maximised'
    xp(langs=langs, train=train, xpNum=xpNum, title='{0} {1} POS {2} Dense1 {3} init {4} {5}'.
       format('Lemma' if useLemma else 'Token', tokenEmb, posEmb, DenseUnitNum, initType, maximisation))


def initImpactTotal():
    desactivateMainConf()
    initImpact(200, 96, 384, initType="frWac200")
    initImpact(200, 96, 384, initType="frWac200", useLemma=True)
    configuration["model"]["embedding"]["frequentTokens"] = False
    initImpact(200, 96, 384, initType="frWac200")
    initImpact(200, 96, 384, initType="frWac200", useLemma=True)

    configuration["model"]["embedding"]["frequentTokens"] = True

    initImpact(250, 128, 384, initType="dataFR.profiles.min.250")
    initImpact(250, 128, 512, initType="dataFR.profiles.min.250", useLemma=True)
    configuration["model"]["embedding"]["frequentTokens"] = False
    initImpact(250, 128, 384, initType="dataFR.profiles.min.250")
    initImpact(250, 128, 512, initType="dataFR.profiles.min.250", useLemma=True)


def initImpactTotal2():
    desactivateMainConf()
    initImpact(200, 56, 32, initType="frWac200", useLemma=True)
    initImpact(200, 64, 64, initType="frWac200", useLemma=True)

    initImpact(250, 56, 32, initType="dataFR.profiles.min.250", useLemma=True)
    initImpact(250, 64, 64, initType="dataFR.profiles.min.250", useLemma=True)


def exploreFRMax(initSeed=0):
    desactivateMainConf()
    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["lemma"] = True
    embConf["usePos"] = True
    embConf["tokenEmb"] = 200
    embConf["posEmb"] = 56
    embConf["initialisation"]["active"] = True
    dense1 = configuration["model"]["mlp"]["dense1"]
    dense1["active"] = True
    dense1["unitNumber"] = 256
    xp(langs=['FR'], train=train, xpNum=xpNum, initSeed=initSeed)


def exploreAllLangs(shuffle=False):
    configuration["xp"]["compo"] = False
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


def exploreRnn(useDense=True):
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


def xpMinimal(title='', initSeed=0, enableInit=False):
    desactivateMainConf()
    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["lemma"] = True
    embConf["usePos"] = True
    embConf["tokenEmb"] = 50
    embConf["posEmb"] = 25
    if enableInit:
        initConf = configuration["model"]["embedding"]["initialisation"]
        initConf["active"] = True
        initConf["type"] = 'frWiki50'

    dense1 = configuration["model"]["mlp"]["dense1"]
    dense1["active"] = True
    dense1["unitNumber"] = 24

    xp(langs=langs, train=train, xpNum=xpNum, title=title, initSeed=initSeed)


def exploreSampling():
    configuration["xp"]["compo"] = False
    configuration["evaluation"]["shuffleTrain"] = False
    # fcDomain = [1, 2, 5, 10]
    fcOSDomain = [2, 5, 10]
    # xpSampling(train=train, title='referencial')
    #
    # xpSampling(train=train, impTrans=True)
    #
    # for fc in fcDomain:
    #     xpSampling(train=train, impTrans=True, FC=fc)
    #
    # xpSampling(train=train, impTrans=True, OS=True)
    #
    # for fc in fcOSDomain:
    #     xpSampling(train=train, impTrans=True, OS=True, FC=fc)
    #
    # xpSampling(train=train, impSent=True)
    #
    # for fc in fcDomain:
    #     xpSampling(train=train, impSent=True, FC=fc)
    #
    # xpSampling(train=train, impSent=True, OS=True)
    #
    # for fc in fcOSDomain:
    #     xpSampling(train=train, impSent=True, OS=True, FC=fc)

    xpSampling(OS=True)

    for fc in fcOSDomain:
        xpSampling(OS=True, FC=fc)


def xpSampling(impSent=False, impTrans=False, OS=False, FC=0, title=''):
    title += 'impSent ' if impSent else ''
    title += 'impTrans ' if impTrans else ''
    title += 'overSampling ' if OS else ''
    title += 'favorisationCoeff = {0}'.format(FC) if FC else ''

    configuration["sampling"]["overSampling"] = OS
    configuration["sampling"]["favorisationCoeff"] = FC
    configuration["sampling"]["sampleWeight"] = True if FC else False

    configuration["sampling"]["importantSentences"] = impSent
    configuration["sampling"]["importantTransitions"] = impTrans
    xpMinimal(title=title)
    configuration["sampling"]["importantTransitions"] = False
    configuration["sampling"]["importantSentences"] = False


def exploreLearning():
    configuration["sampling"]["importantSentences"] = True
    configuration["sampling"]["overSampling"] = True
    configuration["sampling"]["favorisationCoeff"] = 10
    configuration["sampling"]["sampleWeight"] = True
    lrDomain = dict()
    lrDomain['sgd'] = [0.005, 0.01, 0.02, 0.05, 0.1]
    lrDomain['rmsprop'] = [0.0005, 0.001, 0.002, 0.005, 0.01]
    lrDomain['adam'] = [0.0005, 0.001, 0.002, 0.005, 0.01]
    lrDomain['adagrad'] = [0.005, 0.01, 0.02, 0.05, 0.1]
    lrDomain['adadelta'] = [0.1, 0.5, 1.0, 2, 5]
    lrDomain['adamax'] = [0.001, 0.002, 0.005, 0.01]
    lrDomain['nadam'] = [0.001, 0.002, 0.005, 0.01]
    trainConf = configuration["model"]["train"]
    for opt in ['sgd', 'rmsprop', 'adam', 'adagrad', 'adadelta', 'adamax', 'nadam']:
        trainConf["optimizer"] = opt
        for lr in lrDomain[opt]:
            trainConf["lr"] = lr
            xpMinimal(title='')


def exploreLearning2():
    lrDomain = dict()
    # lrDomain['sgd'] = [0.001, 0.0008, 0.0005, 0.0001]
    # lrDomain['sgd'] = [0.005]
    lrDomain['adadelta'] = [0.5, 0.05, 0.02, 0.01, 0.005]
    lrDomain['adamax'] = [0.002, 0.025, 0.05, 0.08, 0.1]
    trainConf = configuration["model"]["train"]
    for opt in ['adadelta', 'adamax']:
        trainConf["optimizer"] = opt
        for lr in lrDomain[opt]:
            trainConf["lr"] = lr
            xpMinimal()


def expolreFavorisationCoeff():
    configuration["sampling"]["importantSentences"] = True
    configuration["sampling"]["overSampling"] = False
    configuration["sampling"]["sampleWeight"] = True

    for coeff in [25, 50, 75, 90, 115, 140]:
        configuration["sampling"]["favorisationCoeff"] = coeff
        xpMinimal()


def setOptimalParameters():
    samling = configuration["sampling"]
    samling["importantSentences"] = True
    samling["overSampling"] = True
    samling["sampleWeight"] = True
    samling["favorisationCoeff"] = 10

    configuration["model"]["train"]["optimizer"] = 'adagrad'
    configuration["model"]["train"]["lr"] = 0.02

    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["usePos"] = True
    embConf["lemma"] = True
    embConf["posEmb"] = 15
    embConf["tokenEmb"] = 200
    setDense1Conf(unitNumber=25)


def dautresXps1():
    setOptimalParameters()
    xp(langs=langs, train=train, cv=cv, xpNum=xpNum)

    setDense1Conf(unitNumber=20)
    xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='Dense 20')

    setDense1Conf(unitNumber=15)
    xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='Dense 15')

    setDense1Conf(unitNumber=10)
    xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='Dense 10')


def dautresXps2():
    setOptimalParameters()
    setDense1Conf(active=False)
    # xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='No Dense')

    setOptimalParameters()
    initConf = configuration["model"]["embedding"]["initialisation"]
    initConf["active"] = True
    initConf["type"] = "dataFR.profiles.min.250"
    embConf = configuration["model"]["embedding"]
    embConf["tokenEmb"] = 250
    xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='Init selvio')
    initConf["type"] = "frWac200"
    embConf["tokenEmb"] = 200
    xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='Init fauconnier')
    initConf["active"] = False


def dautresXps3():
    setOptimalParameters()

    configuration["sampling"]["favorisationCoeff"] = 15
    xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='favorisationCoeff 15')

    configuration["sampling"]["favorisationCoeff"] = 20
    xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='favorisationCoeff 20')

    configuration["sampling"]["favorisationCoeff"] = 30
    xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='favorisationCoeff 30')


def FTB(train=False, corpus=False, useAdam=False, epochs=40, focusedSampling=False,compactVocab=False, sampling=True):
    configuration["dataset"]["FTB"] = True
    setOptimalParameters()
    if useAdam:
        configuration["model"]["train"]["optimizer"] = 'adam'
        configuration["model"]["train"]["lr"] = 0.01
    if not sampling:
        samling = configuration["sampling"]
        samling["overSampling"] = False
        samling["sampleWeight"] = False

    if focusedSampling:
        configuration["sampling"]["focused"] = False
        configuration["sampling"]["mweRepeition"] = 40
    if compactVocab:
        configuration["model"]["embedding"]["compactVocab"] = True
    configuration["model"]["train"]["epochs"] = epochs
    xp(['FR'], train=train, corpus=corpus)
    # for tokEmb in [50, 100, 200]:
    #     configuration["model"]["embedding"]["tokenEmb"] = tokEmb
    #     xp(['FR'], train=True)
    #
    # configuration["model"]["embedding"]["tokenEmb"] = 50
    #
    # samling = configuration["sampling"]
    # samling["sampleWeight"] = False
    # xp(['FR'], train=True)
    # samling["overSampling"] = False
    # xp(['FR'], train=True)


def allLangs(languages, sharedtask2=False, train=False, lemmaEmb=200):
    configuration["dataset"]["sharedtask2"] = sharedtask2
    setOptimalParameters()
    configuration["model"]["embedding"]["tokenEmb"] = lemmaEmb
    xp(languages, corpus=True, train=train)


def runKiperwasser(train=False, epochs=3):
    configuration["xp"]["kiperwasser"] = True
    configuration["model"]["train"]["epochs"] = epochs
    setOptimalParameters()
    xp(['FR'], train=train)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    runKiperwasser(train=False, epochs=2)
    # allLangs(allSharedtask1Lang, sharedtask2=False)
    # allLangs(allSharedtask2Lang, sharedtask2=True)
    # allLangs(['FR'], train=True, sharedtask2=False)
    # FTB(corpus=True)
