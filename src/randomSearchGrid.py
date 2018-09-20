import math
import os
import pickle
import random
import sys

from config import configuration
from identification import xp


def runRandomSearchGridXps(langs, xpNum1=25, sharedtask2=True):
    randomSearchGridPath = os.path.join(configuration["path"]["projectPath"], 'ressources', "randomSearchGrid.p")
    configuration["dataset"]["sharedtask2"] = sharedtask2
    resultDic = pickle.load(open(randomSearchGridPath, "rb"))
    choosenXps, idx = set(), 0
    while idx < xpNum1:
        choosenConfigIdx = random.randint(0, len(resultDic))
        choosenConfigKey = resultDic.keys()[choosenConfigIdx]
        if not resultDic[choosenConfigKey]:
            choosenXps.add(choosenConfigKey)
            resultDic[choosenConfigKey] = '-'
            idx += 1
    pickle.dump(resultDic, open(randomSearchGridPath, "w+b"))
    for k in choosenXps:
        sys.stdout.write(k + '\n')
        setConfig(k)
        configuration['evaluation']['fixedSize'] = True
        xp(langs)
        configuration['evaluation']['fixedSize'] = False
        configuration['evaluation']['train'] = True
        xp(langs)
        configuration['evaluation']['train'] = False
        resultDic = pickle.load(open(randomSearchGridPath, "rb"))
        resultDic[k] = True
        pickle.dump(resultDic, open(randomSearchGridPath, "w+b"))


def createRandomSearchGrid(minimal=False):
    resultDic = dict()
    for i in range(1000):
        if minimal:
            resultDic['_'.join(generateMinimalConfig())] = False
        else:
            resultDic['_'.join(generateConfig())] = False

    pickle.dump(resultDic,
                open(os.path.join(configuration["path"]["projectPath"], 'ressources', "randomSearchGrid.p"),
                     "wb"))
    for k in resultDic:
        print k.replace('_', ',')


def generateConfig():
    sampConf = configuration["sampling"]
    sampConf["favorisationCoeff"] = int(generateValue([0, 100], continousPlage=True))
    sampConf["focused"] = generateValue([True, False])

    embConf = configuration["model"]["embedding"]
    embConf["lemma"] = generateValue([True, False])
    embConf["tokenEmb"] = int(generateValue([20, 500], continousPlage=True))
    embConf["posEmb"] = int(generateValue([5, 200], continousPlage=True))
    vocabType = generateValue(['all', 'frequent', 'compact'])
    embConf["frequent"] = False if vocabType == 'all' else True
    embConf["compactVocab"] = True if vocabType == 'compact' else False

    dense1Conf = configuration["model"]["mlp"]["dense1"]
    dense2Conf = configuration["model"]["mlp"]["dense2"]
    dense1Conf["active"] = False
    dense2Conf["active"] = False
    # if dense1Conf["unitNumber"] > 10:
    dense1Conf["active"] = generateValue([True, False])
    dense1Conf["unitNumber"] = int(generateValue([0, 512], continousPlage=True))
    dense1Conf["activation"] = 'relu'  # generateValue(['relu'])
    dense1Conf["dropout"] = round(generateValue([.0, .6], continousPlage=True), 3)
    dense2Conf['active'] = generateValue([True, False])
    #    if dense2Conf['active']:
    dense2Conf["dropout"] = round(generateValue([.0, .6], continousPlage=True), 3)
    dense2Conf["activation"] = 'relu'  # generateValue(['relu'])
    dense2Conf['unitNumber'] = int(generateValue([0, 128], continousPlage=True))

    trainConf = configuration["model"]["train"]
    trainConf["optimizer"] = 'adagrad'  # generateValue(['adagrad'])
    trainConf["lr"] = round(generateValue([.001, 2], continousPlage=True), 3)
    trainConf["batchSize"] = int(generateValue([16, 256], continousPlage=True))
    trainConf["epochs"] = int(generateValue([30, 100], continousPlage=True))

    res = [str(sampConf["favorisationCoeff"]),
           str(sampConf["focused"]),
           str(embConf["lemma"]),
           str(embConf["tokenEmb"]),
           str(embConf["posEmb"]),
           str(vocabType),
           str(dense1Conf["active"]),
           str(dense1Conf["unitNumber"]),
           str(dense1Conf["activation"]),
           str(dense1Conf["dropout"]),
           str(dense2Conf['active']),
           str(dense2Conf["unitNumber"]),
           str(dense2Conf["activation"]),
           str(dense2Conf["dropout"]),
           str(trainConf["optimizer"]),
           str(trainConf["lr"]),
           str(trainConf["batchSize"]),
           str(trainConf["epochs"])]
    return res


def generateMinimalConfig():
    sampConf = configuration["sampling"]
    sampConf["favorisationCoeff"] = int(generateValue([5, 35], continousPlage=True))
    sampConf["focused"] = True  # generateValue([True, False])

    embConf = configuration["model"]["embedding"]
    embConf["lemma"] = True  # generateValue([True, False])
    embConf["tokenEmb"] = int(generateValue([150, 500], continousPlage=True))
    embConf["posEmb"] = int(generateValue([25, 150], continousPlage=True))
    vocabType = generateValue(['frequent', 'compact'])
    embConf["frequent"] = False if vocabType == 'all' else True
    embConf["compactVocab"] = True if vocabType == 'compact' else False

    dense1Conf = configuration["model"]["mlp"]["dense1"]
    dense2Conf = configuration["model"]["mlp"]["dense2"]
    dense1Conf["active"] = True  # generateValue([True, False])
    dense1Conf["unitNumber"] = int(generateValue([5, 75], continousPlage=True))
    dense1Conf["activation"] = 'relu'  # generateValue(['relu'])
    dense1Conf["dropout"] = round(generateValue([.2, .6], continousPlage=True), 3)
    dense2Conf['active'] = False
    trainConf = configuration["model"]["train"]
    trainConf["optimizer"] = 'adagrad'  # generateValue(['adagrad'])
    trainConf["lr"] = round(generateValue([.01, 0.09], continousPlage=True), 3)
    trainConf["batchSize"] = 128  # int(generateValue([64, 256], continousPlage=True))
    trainConf["epochs"] = 40

    res = [str(sampConf["favorisationCoeff"]),
           str(sampConf["focused"]),
           str(embConf["lemma"]),
           str(embConf["tokenEmb"]),
           str(embConf["posEmb"]),
           str(vocabType),
           str(dense1Conf["active"]),
           str(dense1Conf["unitNumber"]),
           str(dense1Conf["activation"]),
           str(dense1Conf["dropout"]),
           str(dense2Conf['active']),
           str(dense2Conf["unitNumber"]),
           str(dense2Conf["activation"]),
           str(dense2Conf["dropout"]),
           str(trainConf["optimizer"]),
           str(trainConf["lr"]),
           str(trainConf["batchSize"]),
           str(trainConf["epochs"])]
    return res


def generateConfigWithFavorisation():
    sampConf = configuration["sampling"]
    sampConf["favorisationCoeff"] = int(generateValueWithFavorisation([0, 100], [5, 15], continousPlage=True))
    sampConf["focused"] = generateValueWithFavorisation([True, False], [False])

    embConf = configuration["model"]["embedding"]
    embConf["lemma"] = generateValueWithFavorisation([True, False], [True])
    embConf["tokenEmb"] = int(generateValueWithFavorisation([20, 500], [75, 250], continousPlage=True))
    embConf["posEmb"] = int(generateValueWithFavorisation([5, 200], [10, 50], continousPlage=True))
    vocabType = generateValueWithFavorisation(['all', 'frequent', 'compact'], ['frequent'])
    embConf["frequent"] = False if vocabType == 'all' else True
    embConf["compactVocab"] = True if vocabType == 'compact' else False

    dense1Conf = configuration["model"]["mlp"]["dense1"]
    dense2Conf = configuration["model"]["mlp"]["dense2"]
    dense1Conf["active"] = False
    dense2Conf["active"] = False
    dense1Conf["unitNumber"] = int(generateValueWithFavorisation([0, 512], [5, 48], continousPlage=True))
    if dense1Conf["unitNumber"] > 10:
        dense1Conf["active"] = True
        dense1Conf["activation"] = generateValueWithFavorisation(['relu', 'tanh', 'sigmoid'], ['relu'])
        dense1Conf["dropout"] = round(generateValueWithFavorisation([.0, .6], [.2, .5], continousPlage=True), 3)
        dense2Conf['active'] = generateValueWithFavorisation([True, False], [False])
        if dense2Conf['active']:
            dense2Conf["dropout"] = round(generateValueWithFavorisation([.0, .6], [.2, .5], continousPlage=True), 3)
            dense2Conf["activation"] = generateValueWithFavorisation(['relu', 'tanh', 'sigmoid'], ['relu'])
            dense2Conf['unitNumber'] = int(generateValueWithFavorisation([0, 128], [8, 24], continousPlage=True))

    trainConf = configuration["model"]["train"]
    trainConf["optimizer"] = generateValueWithFavorisation(['adagrad'], ['adagrad'])
    # trainConf["optimizer"] = generateValue(['adam', 'adagrad'], ['adagrad'])
    trainConf["lr"] = generateValueWithFavorisation([.0001, 1], [.005, .05], continousPlage=True) \
        if trainConf["optimizer"] == 'adam' \
        else generateValueWithFavorisation([.001, 2], [.005, .1], continousPlage=True)
    trainConf["batchSize"] = int(generateValueWithFavorisation([16, 256], [48, 80], continousPlage=True))

    res = [str(sampConf["favorisationCoeff"]),
           str(sampConf["focused"]),
           str(embConf["lemma"]),
           str(embConf["tokenEmb"]),
           str(embConf["posEmb"]),
           str(vocabType),
           str(dense1Conf["active"]),
           str(dense1Conf["unitNumber"]),
           str(dense1Conf["activation"]),
           str(dense1Conf["dropout"]),
           str(dense2Conf['active']),
           str(dense2Conf["unitNumber"]),
           str(dense2Conf["activation"]),
           str(dense2Conf["dropout"]),
           str(trainConf["optimizer"]),
           str(trainConf["lr"]),
           str(trainConf["batchSize"])]
    return res


def generateValue(plage, continousPlage=False, uniform=False, favorisationTaux=0.7):
    if continousPlage:
        if uniform:
            return random.uniform(plage[0], plage[-1])
        else:
            return pow(2, random.uniform(math.log(plage[0], 2), math.log(plage[-1], 2)))
    else:
        if not uniform:
            alpha = random.uniform(0, 1)
            if alpha < favorisationTaux:
                return plage[0]
            return plage[random.randint(1, len(plage) - 1)]
        else:
            return plage[random.randint(0, len(plage) - 1)]


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


def setConfig(line):
    sampConf = configuration["sampling"]
    embConf = configuration["model"]["embedding"]
    trainConf = configuration["model"]["train"]
    dense1Conf = configuration["model"]["mlp"]["dense1"]
    dense2Conf = configuration["model"]["mlp"]["dense2"]

    values = line.split('_')

    sampConf["favorisationCoeff"] = int(values[0])
    sampConf["focused"] = values[1] == 'True'
    embConf["lemma"] = values[2] == 'True'
    embConf["tokenEmb"] = int(values[3])
    embConf["posEmb"] = int(values[4])
    vocabType = values[5]
    embConf["frequent"] = False if vocabType == 'all' else True
    embConf["compactVocab"] = True if vocabType == 'compact' else False
    dense1Conf["active"] = values[6] == 'True'
    dense1Conf["unitNumber"] = int(values[7])
    dense1Conf["activation"] = values[8]
    dense1Conf["dropout"] = round(float(values[9]), 3)

    dense2Conf['active'] = values[10] == 'True' and int(values[11]) > 0
    dense2Conf["unitNumber"] = int(values[11])
    dense2Conf["activation"] = values[12]
    dense2Conf["dropout"] = round(float(values[13]), 3)
    trainConf["optimizer"] = values[14]
    trainConf["lr"] = round(float(values[15]), 4)
    trainConf["batchSize"] = int(values[16])
    trainConf["epochs"] = int(values[17])


def test():
    importantV, nonImportantV = 0, 0
    for i in range(10000):
        # v = generateValue([5, 20], [8, 12], continousPlage=True)
        # v = generateValue(['A', 'B', 'C', 'D', 'E', 'F', 'G'], ['B', 'C', 'D'], continousPlage=False)
        v = generateValue([1, 6, 4, 5, 66, 88, 444], continousPlage=False)
        # print v
        if v in [5, 4]:  # ['B', 'C', 'D']:
            # if 12 > v > 8:
            importantV += 1
        else:
            nonImportantV += 1
    print importantV, nonImportantV


def test2():
    x1, x2 = [], []
    for i in range(1000):
        x2.append(generateValueWithFavorisation([18, 1024], [18, 128], 70, continousPlage=True))
        x1.append(generateValue([18, 1024], continousPlage=True, uniform=False))
    import matplotlib.pyplot as plt

    plt.plot(x1, 'ro')
    plt.ylabel('Drawn geometrically')
    plt.show()

    plt.plot(x2, 'ro')
    plt.ylabel('Zone prioritaire')
    plt.show()
