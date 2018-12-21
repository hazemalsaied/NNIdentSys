import math

from corpus import *
from xpTools import xp, XpMode


def runRSG(langs, dataset, xpMode, division, fileName, xpNumByThread=50, xpNum=1, mlpInLinear=False, linearInMlp=False,
           complentary=False):
    for i in range(xpNumByThread):
        exps = getGrid(fileName)
        while True:
            xpIdx = random.randint(1, len(exps)) - 1
            exp = exps.keys()[xpIdx]
            if not exps[exp][0]:
                break
        exps[exp][0] = True
        pickle.dump(exps, open(os.path.join(configuration['path']['projectPath'], 'ressources/RSG', fileName), 'wb'))
        configuration.update(exps[exp][1])
        xp(langs, dataset, xpMode, division, xpNum=xpNum,
           mlpInLinear=mlpInLinear, linearInMlp=linearInMlp,
           complentary=complentary)


def createRSG(fileName, xpMode, xpNum=1000):
    resultDic = dict()
    for i in range(xpNum):
        generateConf(xpMode)
        resultDic[i] = [False, copy.deepcopy(configuration)]
        # resultDic[0] == resultDic[1]:
    pickle.dump(resultDic, open(os.path.join(configuration['path']['projectPath'],
                                             'ressources/RSG', fileName), 'wb'))


def getGrid(fileName):
    randomSearchGridPath = os.path.join(configuration['path']['projectPath'], 'ressources/RSG', fileName)
    return pickle.load(open(randomSearchGridPath, 'rb'))


def generateConf(xpMode):
    if xpMode == XpMode.rnn:
        generateRNNConf()
    elif xpMode == XpMode.linear:
        generateLinearConf()
    elif xpMode == XpMode.kiperwasser:
        generateKiperwasserConf()
    else:
        generateMLPConf()


def generateMLPConf():
    configuration['mlp'].update({
        'posEmb': int(generateValue([15, 150], uniform=False, continousPlage=True)),
        'tokenEmb': int(generateValue([100, 600], uniform=False, continousPlage=True)),
        'compactVocab': generateValue([True, False], uniform=True),
        'trainable': generateValue([True, False], continousPlage=False, uniform=False, favorisationTaux=0.8),
        'dense1UnitNumber': int(generateValue([25, 600], uniform=False, continousPlage=True)),
        'dense1Dropout': float(generateValue([.1, .2, .3, .4, .5, .6], continousPlage=False, uniform=True)),
        'dense1Activation': generateValue(['relu', 'tanh'], continousPlage=False, uniform=False, favorisationTaux=0.8),
        'lemma': generateValue([True, False], continousPlage=False, uniform=False),
        'batchSize': generateValue([16, 32, 48, 64, 128], continousPlage=False, uniform=True),
        'lr': round(generateValue([.01, .2], continousPlage=True, uniform=False), 3)
    })
    configuration['sampling'].update({
        'overSampling': True,
        'importantSentences': True,
        'importantTransitions': False,
        'sampleWeight': generateValue([True, False], continousPlage=False, uniform=True),
        'favorisationCoeff': int(generateValue([1, 40], continousPlage=True, uniform=False)),
        'focused': generateValue([True, False], continousPlage=False, uniform=True),
    })
    configuration['others'].update({
        'mweRepeition': generateValue([5, 10, 15, 20, 30], continousPlage=False, uniform=True),
    })


def generateRNNConf():
    configuration['rnn']['wordDim'] = int(generateValue([250, 600], True, True))
    configuration['rnn']['posDim'] = int(generateValue([25, 100], True, True))

    configuration['rnn']['gru'] = True  # generateValue([True, False], False, False)
    configuration['rnn']['wordRnnUnitNum'] = int(generateValue([25, 200], True, True))
    configuration['rnn']['posRnnUnitNum'] = int(generateValue([25, 200], True, True))
    configuration['rnn']['rnnDropout'] = round(generateValue([0, .3], True, True), 1)

    configuration['rnn']['useDense'] = True  # generateValue([True, False], False, False)
    configuration['rnn']['denseDropout'] = 0  # round(generateValue([0, .5], True, True), 1)
    configuration['rnn']['denseUnitNum'] = int(generateValue([5, 200], True, True))
    configuration['rnn']['batchSize'] = 64  # generateValue([16, 32, 64, 128], False, True)
    configuration['mlp']['compactVocab'] = generateValue([True, False], False, False)


def generateLinearConf():
    configuration['features'].update({
        'lemma': True,
        'token': generateValue([True, False], favorisationTaux=.3),
        'pos': True,
        'suffix': False,
        'b1': generateValue([True, False], favorisationTaux=.5),
        'bigram': True,
        's0b2': generateValue([True, False], favorisationTaux=.5),
        'trigram': generateValue([True, False], favorisationTaux=.5),
        'syntax': False,
        'syntaxAbstract': False,
        'dictionary': generateValue([True, False], favorisationTaux=.5),
        's0TokenIsMWEToken': generateValue([True, False], favorisationTaux=.5),
        's0TokensAreMWE': False,
        'history1': generateValue([True, False], favorisationTaux=.5),
        'history2': generateValue([True, False], favorisationTaux=.5),
        'history3': generateValue([True, False], favorisationTaux=.5),
        'stackLength': generateValue([True, False], favorisationTaux=.5),
        'distanceS0s1': generateValue([True, False], favorisationTaux=.5),
        'distanceS0b0': generateValue([True, False], favorisationTaux=.5)
    })


def generateKiperwasserConf():
    kiperConf = configuration['kiperwasser']
    kiperConf['wordDim'] = int(generateValue([50, 500], True))
    kiperConf['posDim'] = int(generateValue([15, 150], True))
    kiperConf['denseActivation'] = 'tanh'  # str(generateValue(['tanh', 'relu'], False))
    kiperConf['optimizer'] = 'adagrad'  # str(generateValue(['adam', 'adagrad'], False))
    kiperConf['lr'] = 0.07
    kiperConf['dense1'] = int(generateValue([10, 350], True))
    kiperConf['rnnDropout'] = round(generateValue([.1, .4], True), 2)
    kiperConf['rnnUnitNum'] = int(generateValue([20, 250], True))
    kiperConf['rnnLayerNum'] = 1  # generateValue([1, 2], False)

    configuration['mlp']['compactVocab'] = generateValue([True, False], False)
    configuration['mlp']['lemma'] = True  # generateValue([True, False], False)


def generateValue(plage, continousPlage=False, uniform=False, favorisationTaux=0.7):
    if continousPlage:
        if uniform:
            return random.uniform(plage[0], plage[-1])
        else:
            return pow(2, random.uniform(math.log(plage[0], 2), math.log(plage[-1], 2)))
    else:
        if not uniform and len(plage) == 2:
            alpha = random.uniform(0, 1)
            if alpha < favorisationTaux:
                return plage[0]
            return plage[random.randint(1, len(plage) - 1)]
        else:
            return plage[random.randint(0, len(plage) - 1)]


def hasCloseValue(value, generatedValues, step=0.001):
    for v in generatedValues:
        if abs(v - value) <= step:
            return True
    return False


def generateValueWithFavorisation(plage, importantPlage, favorisationRate=60, continousPlage=False):
    useImportantPlage = random.randint(0, 100) < favorisationRate
    if useImportantPlage:
        if continousPlage:
            return random.uniform(importantPlage[0], importantPlage[-1])
        else:
            return importantPlage[random.randint(0, len(importantPlage) - 1)]
    else:
        if continousPlage:
            leftRange = [plage[0], importantPlage[0]]
            leftRangeLegth = importantPlage[0] - plage[0]
            rightRange = [importantPlage[-1], plage[-1]]
            rightRangeLenght = plage[-1] - importantPlage[-1]
            percentage = (float(leftRangeLegth) / (leftRangeLegth + rightRangeLenght))
            useLeftRange = random.uniform(0, 1) < percentage
            return random.uniform(leftRange[0], leftRange[-1]) if useLeftRange else \
                random.uniform(rightRange[0], rightRange[-1])

        else:
            for item in importantPlage:
                plage.remove(item)
            return plage[random.randint(0, len(plage) - 1)]


def createLRGrid(xpNum=150):
    lrs = dict()
    for i in range(xpNum):
        while True:
            lr = round(generateValue([.001, .2], True, True), 3)
            if lr not in lrs.keys() and not hasCloseValue(lr, lrs):
                lrs[lr] = False
                break
    pickle.dump(lrs, open(os.path.join(configuration['path']['projectPath'],
                                       'ressources/RSG', 'lrGrid.p'), 'wb'))
