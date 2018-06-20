from config import *
from identification import xp


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


def exploreEmbImpact(tokenEmbs, posEmbs=None, denseDomain=None, useLemma=False, usePos=False, train=False, cv=False,
                     xpNum=10, title='',
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


def initImpact(tokenEmb, posEmb, DenseUnitNum, initType="frWac200", useLemma=False, train=True, xpNum=10, langs=['FR']):
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


def exploreFRMax(train=False, xpNum=10, initSeed=0):
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


def exploreRnn(train=True, useDense=True):
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


def xpMinimal(train=False, xpNum=10, title='', initSeed=0):
    desactivateMainConf()
    embConf = configuration["model"]["embedding"]
    embConf["active"] = True
    embConf["lemma"] = True
    embConf["usePos"] = True
    embConf["tokenEmb"] = 48
    embConf["posEmb"] = 24
    dense1 = configuration["model"]["mlp"]["dense1"]
    dense1["active"] = True
    dense1["unitNumber"] = 24
    xp(langs=['FR'], train=train, xpNum=xpNum, title=title, initSeed=initSeed)


def exploreSampling(train=True):
    configuration["xp"]["compo"] = False
    configuration["evaluation"]["shuffleTrain"] = False
    fcDomain = [1, 2, 5, 10]
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

    xpSampling(train=train, OS=True)

    for fc in fcOSDomain:
        xpSampling(train=train, OS=True, FC=fc)


def xpSampling(train=True, xpNum=10, impSent=False, impTrans=False, OS=False, FC=0, title=''):
    title += 'impSent ' if impSent else ''
    title += 'impTrans ' if impTrans else ''
    title += 'overSampling ' if OS else ''
    title += 'favorisationCoeff = {0}'.format(FC) if FC else ''

    configuration["model"]["train"]["sampling"]["overSampling"] = OS
    configuration["model"]["train"]["favorisationCoeff"] = FC
    configuration["model"]["train"]["sampleWeight"] = True if FC else False

    configuration["model"]["train"]["sampling"]["importantSentences"] = impSent
    configuration["model"]["train"]["sampling"]["importantTransitions"] = impTrans
    xpMinimal(train=train, xpNum=xpNum, title=title)
    configuration["model"]["train"]["sampling"]["importantTransitions"] = False
    configuration["model"]["train"]["sampling"]["importantSentences"] = False


def exploreLearning(train=True, xpNum=5):
    configuration["model"]["train"]["sampling"]["importantSentences"] = True
    configuration["model"]["train"]["sampling"]["overSampling"] = True
    configuration["model"]["train"]["favorisationCoeff"] = 10
    configuration["model"]["train"]["sampleWeight"] = True
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
            xpMinimal(train=train, xpNum=xpNum, title='')


def exploreLearning2(train=True, xpNum=5):
    lrDomain = dict()
    lrDomain['sgd'] = [0.001, 0.0008, 0.0005, 0.0001]
    lrDomain['adadelta'] = [0.05, 0.02, 0.01, 0.005]
    lrDomain['adamax'] = [0.02, 0.05, 0.08, 0.1]
    trainConf = configuration["model"]["train"]
    for opt in ['sgd', 'adagrad', 'adamax']:
        trainConf["optimizer"] = opt
        for lr in lrDomain[opt]:
            trainConf["lr"] = lr
            xpMinimal(train=train, xpNum=xpNum, title='')


def expolreFavorisationCoeff():
    configuration["model"]["train"]["sampling"]["importantSentences"] = True
    configuration["model"]["train"]["sampling"]["overSampling"] = False
    for coeff in [15, 25, 40, 50, 75, 90, 125]:
        configuration["model"]["train"]["favorisationCoeff"] = coeff
        xpMinimal(True, xpNum=5)


if __name__ == '__main__':
    configuration["model"]["train"]["sampling"]["importantSentences"] = True
    configuration["model"]["train"]["sampling"]["overSampling"] = True
    configuration["model"]["train"]["sampleWeight"] = True
    configuration["model"]["train"]["favorisationCoeff"] = 10

    # parser = argparse.ArgumentParser(description='Process some xps.')

    # parser.add_argument('xpLbl')
    # args = parser.parse_args()
    # xp1, xp2, xp3, xp4, xp5, xp6, xp7, xp8 = [30], [40], [50], [60], [70], [100], [125], [150]
    # xpDic = {
    #     'xp1': [30], 'xp2': [40], 'xp3': [50], 'xp4': [60],
    #     'xp5': [70], 'xp6': [100], 'xp7': [125], 'xp8': [150]
    # }
    # if args.xpLbl == 'learning2':
    #    print 'learning2'
    exploreLearning2(train=True)

    # if args.xpLbl.startswith('xp'):
    #    if args.xpLbl in xpDic:
    # print args.xpLbl
    # exploreEmbImpact(xpDic[args.xpLbl], [15, 25, 35, 50], [25, 75, 125, 250],
    #                 useLemma=True, usePos=True, train=False, xpNum=5, langs=['FR'])
    # for dom in xpDic.values():
    #     exploreEmbImpact(dom, [15, 25, 35, 50], [25, 75, 125, 250],
    #                      useLemma=True, usePos=True, train=False, xpNum=5, langs=['FR'])

    # if args.xpLbl == 'coeff':
    for coeff in [15, 20, 25, 30]:
        configuration["model"]["train"]["favorisationCoeff"] = coeff
        xpMinimal(True, 5)
    # configuration["xp"]["pytorch"] = True
    # xpMinimal(True, 1)
