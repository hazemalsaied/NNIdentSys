import numpy

from config import *
from corpus import *
from identification import identify, crossValidation

allLangs = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT', 'LT', 'MT', 'PL', 'PT', 'RO', 'SL', 'SV', 'TR']


def xp(debug=True, train=False, cv=False, xpNum=10, title='', langs=['FR']):
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
        identify(langs=langs)
    if train:
        ######################################
        #   Train
        ######################################
        evlaConf["debug"] = False
        evlaConf["train"] = True
        sys.stdout.write('Train enabled\n')
        for i in range(xpNum):
            numpy.random.seed(seed)
            random.seed(seed)
            sys.stdout.write('# Seed = {0}\n'.format(seed))
            if title:
                reports.createHeader(title)
            identify(langs=langs)
            seed += 1
        evlaConf["debug"] = True
        evlaConf["train"] = False
    if cv:
        ######################################
        #   CV Debug
        ######################################
        sys.stdout.write('CV debug enabled\n')
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
            sys.stdout.write('CV enabled\n')
            crossValidation(langs=langs)
            ######################################
            #   Load
            ######################################
            # preTrainedPath= '/home/halsaied/nancy/NNIdenSys/NNIdenSys/Reports/FR-12/12-FR-modelWeigth.hdf5'
            # identify(load=configuration["evaluation"]["load"], loadFolderPath=loadFolderPath)


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
                newtitle = 'Tokens({0}) POS({1}) {2}'.format(tokenEmb, posEmb, title)
                xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=newtitle)
        else:
            embConf["tokenEmb"] = tokenEmb
            newtitle = 'Tokens({0}) {1}'.format(tokenEmb, title)
            xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=newtitle)

    embConf["usePos"] = False
    embConf["lemma"] = False


def exploreTokenPosEmbImpact(domain, train=False, cv=False, xpNum=10, usePos=False, langs=['FR']):
    # desactivateMainConf()
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
        xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title=title)


def exploreRnnUnitNum(rnnDomain, train=False, cv=False, xpNum=10, title='', langs=['FR']):
    rnnConf = configuration["model"]["rnn"]["rnn1"]
    rnnConf["active"] = True
    for unitNum in rnnDomain:
        rnnConf["unitNumber"] = unitNum
        xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='{0} + RNN {1}'.format(title, unitNum))
    rnnConf["active"] = False


def xpGRU(stacked=False, gru=False, cv=False, langs=['FR']):
    rnnConf = configuration["model"]["rnn"]
    rnnConf["active"] = True
    rnnConf["gru"] = True
    rnnConf["stacked"] = True
    title = '' if not stacked else 'Stacked '
    title += ' LSTM ' if not gru else ' GRU '
    xp(langs=langs, cv=cv, train=not cv, title=title)
    rnnConf["active"] = False
    rnnConf["gru"] = False
    rnnConf["stacked"] = False


def tokenPOSEmbImpact(langs=['FR']):
    exploreTokenPosEmbImpact([25, 50, 100, 150, 200, 250, 300], train=True, xpNum=10, usePos=False)
    configuration["model"]["embedding"]["tokenEmb"] = 50
    exploreTokenPosEmbImpact([8, 16, 24, 32, 40, 48, 56], train=True, xpNum=10, usePos=True)
    configuration["model"]["embedding"]["tokenEmb"] = 100
    exploreTokenPosEmbImpact([8, 16, 24, 32, 40, 48, 56], train=True, xpNum=10, usePos=True)
    configuration["model"]["embedding"]["tokenEmb"] = 150
    exploreTokenPosEmbImpact([8, 16, 24, 32, 40, 48, 56], train=True, xpNum=10, usePos=True)
    configuration["model"]["embedding"]["tokenEmb"] = 200
    exploreTokenPosEmbImpact([8, 16, 24, 32, 40, 48, 56], train=True, xpNum=10, usePos=True)


def exploreBinaryPOSEmb(domain, train=False, cv=False, xpNum=10, title='', langs=['FR']):
    desactivateMainConf()
    configuration["model"]["embedding"]["active"] = True
    configuration["model"]["embedding"]["usePos"] = True
    configuration["model"]["embedding"]["initialisation"]["active"] = True
    configuration["model"]["embedding"]["initialisation"]["oneHotPos"] = True
    configuration["model"]["embedding"]["initialisation"]["token"] = False

    for embDim in domain:
        configuration["model"]["embedding"]["tokenEmb"] = embDim
        xp(langs=langs, train=train, cv=cv, xpNum=xpNum, title='{0} Token {1}+ Binary POS Emb '.format(title, embDim))


def exploreEarlyStopParams(train=False, cv=False, xpNum=10, langs=['FR']):
    desactivateMainConf()
    setFeatureConf()
    setDense1Conf()
    setEmbConf()
    trainConf = configuration["model"]["train"]
    # monitor :  val_loss
    trainConf["monitor"] = 'val_loss'
    minDeltaDomain = [.3, .2, .1, .05, .01]
    for minDelta in minDeltaDomain:
        trainConf["minDelta"] = minDelta
        xp(langs=langs, train=train, cv=cv, xpNum=xpNum)
    trainConf["monitor"] = 'val_acc'
    for minDelta in minDeltaDomain:
        trainConf["minDelta"] = minDelta
        xp(langs=langs, train=train, cv=cv, xpNum=xpNum)


def tableEmb(useFeatures=False, train=False, cv=False, xpNum=10, langs=['FR']):
    tokenDmain = [25, 50, 75, 100, 125, 150]
    posDomain = [16, 24, 32]
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


def denseImpact(useLemma=False, train=True, langs=['FR']):
    embConf = configuration["model"]["embedding"]
    desactivateMainConf()
    tokenEmbDic = {}
    if useLemma:
        tokenEmbDic[25] = [24, 32]
        tokenEmbDic[50] = [16]
        tokenEmbDic[75] = [24]
    else:
        tokenEmbDic[25] = [8, 16]
        tokenEmbDic[50] = [24]
        tokenEmbDic[75] = [24]
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
                print res + ' {0} POS {1} Dense {2}'.format(tokenEmb, posEmb, denseUnitNum)
                xp(langs=langs, train=train,
                   title=res + ' {0} POS {1} Dense {2}'.format(tokenEmb, posEmb, denseUnitNum))


def initImpact(tokenEmb, posEmb, DenseUnitNum, initType="frWac200", useLemma=False, train=True, langs=['FR']):
    embConf = configuration["model"]["embedding"]
    setEmbConf(tokenEmb=tokenEmb, posEmb=posEmb, usePos=True, init=True)
    embConf["lemma"] = useLemma
    embConf["initialisation"]["type"] = initType  # "frWac200"  # "dataFR.profiles.min.250"
    setDense1Conf(unitNumber=DenseUnitNum)
    maximisation = '' if configuration["model"]["embedding"]["frequentTokens"] else 'Maximised'
    xp(langs=langs, train=train, title='{0} {1} POS {2} Dense1 {3} init {4} {5}'.
       format('Lemma' if useLemma else 'Token', tokenEmb, posEmb, DenseUnitNum, initType, maximisation))


def initImpactTotal(train=False, langs=['FR']):
    desactivateMainConf()
    initImpact(200, 96, 384, initType="frWac200", train=train)
    initImpact(200, 96, 384, initType="frWac200", useLemma=True, train=train)
    configuration["model"]["embedding"]["frequentTokens"] = False
    initImpact(200, 96, 384, initType="frWac200", train=train)
    initImpact(200, 96, 384, initType="frWac200", useLemma=True, train=train)

    configuration["model"]["embedding"]["frequentTokens"] = True

    initImpact(250, 128, 384, initType="dataFR.profiles.min.250", train=train)
    initImpact(250, 128, 512, initType="dataFR.profiles.min.250", useLemma=True, train=train)
    configuration["model"]["embedding"]["frequentTokens"] = False
    initImpact(250, 128, 384, initType="dataFR.profiles.min.250", train=train)
    initImpact(250, 128, 512, initType="dataFR.profiles.min.250", useLemma=True, train=train)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)

    initImpactTotal(train=True, langs=['FR'])

    # exploreInitParams(useLemma=True, train=True)
    # denseImpact(useLemma=True, train=True)

    # tokenDmain = [25, 50, 75, 100, 125, 150]
    # posDomain = [8, 16, 24, 32]
    # desactivateMainConf()
    # setEmbConf(usePos=False, init=False)
    # 1 token + POS
    # exploreEmbImpact(tokenDmain, posDomain, usePos=True, useLemma=False, train=True)
    # 2 Lemma + POS
    # exploreEmbImpact(tokenDmain, posDomain, useLemma=True, usePos=True, train=True)
    # # 3 Lemma + POS + Feature
    # configuration["features"]["active"] = True
    # title = 'Features '
    # exploreEmbImpact(tokenDmain, posDomain, useLemma=True, usePos=True, train=True, title=title)
