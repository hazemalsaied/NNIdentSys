import os
import pickle
import random
import sys

from config import configuration
from identification import xp
from randomSearchGrid import generateValue


def createRSGrid(xpNum=500, fileName='rnnRsgGrid.p'):
    xps = dict()
    for i in range(xpNum):
        configuration['rnn']['wordDim'] = int(generateValue([50, 500], True, True))
        configuration['rnn']['posDim'] = int(generateValue([10, 100], True, True))

        configuration['rnn']['gru'] = generateValue([True, False], False, False)
        configuration['rnn']['wordRnnUnitNum'] = int(generateValue([5, 100], True, True))
        configuration['rnn']['posRnnUnitNum'] = int(generateValue([5, 50], True, True))
        configuration['rnn']['rnnDropout'] = round(generateValue([0, .5], True, True), 1)

        configuration['rnn']['useDense'] = generateValue([True, False], False, False)
        configuration['rnn']['denseDropout'] = round(generateValue([0, .5], True, True), 1)
        configuration['rnn']['denseUnitNum'] = int(generateValue([5, 100], True, True))
        configuration['rnn']['batchSize'] = generateValue([16, 32, 64, 128], False, True)
        configuration['rnn']['compactVocab'] = generateValue([True, False], False, False)
        xps[i] = [False, dict(configuration['rnn'])]
    pickle.dump(xps, open(os.path.join(configuration['path']['projectPath'],
                                       'ressources', fileName), 'wb'))


def getGrid(fileName):
    randomSearchGridPath = os.path.join(configuration['path']['projectPath'], 'ressources', fileName)
    return pickle.load(open(randomSearchGridPath, 'rb'))


def runLrRSG(fileName='lrGrid.p'):
    for i in range(30):
        lrs = getGrid(fileName)
        while True:
            lrIdx = random.randint(1, len(lrs)) - 1
            lr = lrs.keys()[lrIdx]
            if not lrs[lr]:
                break
        lrs[lr] = True
        pickle.dump(lrs, open(os.path.join(configuration['path']['projectPath'], 'ressources', 'lrGrid.p'), 'wb'))
        configuration['rnn']['lr'] = lr
        print lr
        configuration['rnn']['optimizer'] = 'adam'
        sys.stdout.write(str(configuration['rnn']))
        sys.stdout.write('\n# Optimiser: {0} LR: {1}\n'.format(
            configuration['rnn']['optimizer'], configuration['rnn']['lr']))
        xp(langs, xpNum=1)
        configuration['rnn']['optimizer'] = 'adagrad'
        sys.stdout.write('\n# Optimiser: {0} LR: {1}\n'.format(
            configuration['rnn']['optimizer'], configuration['rnn']['lr']))
        xp(langs, xpNum=1)


def runRSG(xpNumByThread=60, fileName='rnnRsgGrid.p', flipLemma=False):
    for i in range(xpNumByThread):
        exps = getGrid(fileName)
        while True:
            xpIdx = random.randint(1, len(exps)) - 1
            exp = exps.keys()[xpIdx]
            if not exps[exp][0]:
                break
        exps[exp][0] = True
        pickle.dump(exps, open(os.path.join(configuration['path']['projectPath'], 'ressources', fileName), 'wb'))
        configuration['rnn'].update(exps[exp][1])
        if flipLemma:
            configuration['model']['embedding']['lemma'] = generateValue([True, False], False, False)
        sys.stdout.write('\n# RNN Conf :' + str(configuration['rnn']) + ', Lemma: {0}\n'.
                         format(configuration['model']['embedding']['lemma']))
        xp(langs, xpNum=1)


def createCloserRSGrid(xpNum=500, fileName='rnnCloserRsgGrid.p'):
    xps = dict()
    for i in range(xpNum):
        configuration['rnn']['wordDim'] = int(generateValue([200, 500], True, True))
        configuration['rnn']['posDim'] = int(generateValue([10, 100], True, True))

        configuration['rnn']['gru'] = True  # generateValue([True, False], False, False)
        configuration['rnn']['wordRnnUnitNum'] = int(generateValue([5, 100], True, True))
        configuration['rnn']['posRnnUnitNum'] = int(generateValue([5, 50], True, True))
        configuration['rnn']['rnnDropout'] = round(generateValue([0, .4], True, True), 1)

        configuration['rnn']['useDense'] = True  # generateValue([True, False], False, False)
        configuration['rnn']['denseDropout'] = round(generateValue([0, .4], True, True), 1)
        configuration['rnn']['denseUnitNum'] = int(generateValue([5, 100], True, True))
        configuration['rnn']['batchSize'] = generateValue([16, 32, 64, 128], False, True)
        configuration['rnn']['compactVocab'] = True  # generateValue([True, False], False, False)
        xps[i] = [False, dict(configuration['rnn'])]
    pickle.dump(xps, open(os.path.join(configuration['path']['projectPath'],
                                       'ressources', fileName), 'wb'))


def hasCloseValue(value, generatedValues, step=0.001):
    for v in generatedValues:
        if abs(v - value) <= step:
            return True
    return False


def createLRGrid(xpNum=150):
    lrs = dict()
    for i in range(xpNum):
        while True:
            lr = round(generateValue([.001, .2], True, True), 3)
            if lr not in lrs.keys() and not hasCloseValue(lr, lrs):
                lrs[lr] = False
                break
    pickle.dump(lrs, open(os.path.join(configuration['path']['projectPath'],
                                       'ressources', 'lrGrid.p'), 'wb'))


def exploreBestConfs():
    configuration['rnn']['gru'] = True
    configuration['rnn']['useDense'] = True
    configuration['rnn']['compactVocab'] = True

    bestConfs = [[283, 25, 94, 0.1, 64, 37, 0, 64],
                 [267, 78, 96, 0.3, 31, 40, 0, 16],
                 [410, 54, 51, 0.1, 25, 15, 0.1, 16]]
    for c in bestConfs:
        configuration['rnn']['wordDim'] = c[0]
        configuration['rnn']['posDim'] = c[1]
        configuration['rnn']['denseUnitNum'] = c[2]
        configuration['rnn']['denseDropout'] = c[3]
        configuration['rnn']['wordRnnUnitNum'] = c[4]
        configuration['rnn']['posRnnUnitNum'] = c[5]
        configuration['rnn']['rnnDropout'] = c[6]
        configuration['rnn']['batchSize'] = c[7]

        xp(langs, xpNum=1)


def exploreExtraSampling():
    configuration['rnn']['gru'] = True
    configuration['rnn']['useDense'] = True
    configuration['rnn']['compactVocab'] = True

    bestConfs = [[410, 54, 51, 0.1, 25, 15, 0.1, 16]]
    # [283, 25, 94, 0.1, 64, 37, 0, 64],
    # [267, 78, 96, 0.3, 31, 40, 0, 16]]
    for c in bestConfs:

        configuration['rnn']['wordDim'] = c[0]
        configuration['rnn']['posDim'] = c[1]
        configuration['rnn']['denseUnitNum'] = c[2]
        configuration['rnn']['denseDropout'] = c[3]
        configuration['rnn']['wordRnnUnitNum'] = c[4]
        configuration['rnn']['posRnnUnitNum'] = c[5]
        configuration['rnn']['rnnDropout'] = c[6]
        configuration['rnn']['batchSize'] = c[7]
        xp(langs, xpNum=1)
        configuration['sampling']['focused'] = True
        xp(langs, xpNum=1)
        configuration['sampling']['sampleWeight'] = True
        for fc in [5, 15, 25]:
            configuration['sampling']['favorisationCoeff'] = fc
            xp(langs, xpNum=3)
        configuration['sampling']['sampleWeight'] = False
        configuration['sampling']['focused'] = False


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    import config
    from identification import setTrainAndTest, setXPMode, setDataSet

    setDataSet(config.Dataset.sharedtask2)
    setTrainAndTest(config.Evaluation.fixedSize)
    setXPMode(config.XpMode.rnn)

    langs = ['BG', 'PT', 'TR']

    configuration['rnn'].update({
        'wordDim': 100,
        'posDim': 25,
        'compactVocab': True,
        'gru': True,
        'wordRnnUnitNum': 25,
        'posRnnUnitNum': 10,
        'rnnDropout': .3,
        'useDense': True,
        'denseDropout': .1,
        'denseUnitNum': 25,
        'optimizer': 'adagrad',
        'lr': .05,
        'epochs': 20,
        'batchSize': 64,
        'earlyStop': True,
        's0TokenNum': 4,
        's1TokenNum': 2,
        'bTokenNum': 1,
        'shuffle': False,
        'rnnSequence': False
    })
    configuration['sampling']['importantSentences'] = True
    configuration['sampling']['overSampling'] = True
    configuration['model']['embedding']['lemma'] = True

    exploreExtraSampling()
    # exploreBestConfs()
    # xp(langs, xpNum=1)

    # runRSG(30, fileName='rnnCloserRsgGrid.p')
    # createCloserRSGrid()
    # createRSGrid()
    # runRSG()
    # langs = ['FR']
    # xp(langs, xpNum=1)
    # langs = ['BG']
    # createLRGrid()
    # runLrRSG(langs)
    # xp(langs, xpNum=1)
