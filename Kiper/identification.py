import datetime
import logging
import math
import random

import torch

import modelCompactKiper
import modelKiperwasser
import oracle
import parserCompact
import reports
from corpus import *
from evaluation import evaluate
from parser import parse


def xp(langs, xpNum=3, title='', compact=False):
    reports.createHeader(title)
    for lang in langs:
        for _ in range(xpNum):
            corpus = Corpus(lang)
            oracle.parse(corpus)
            startTime = datetime.datetime.now()
            network = modelKiperwasser.train(corpus) if not compact else modelCompactKiper.train(corpus)
            sys.stdout.write(reports.doubleSep + reports.tabs + 'Training time : {0}'.
                             format(datetime.datetime.now() - startTime) + reports.doubleSep)
            if compact:
                parserCompact.parse(corpus, network)
            else:
                parse(corpus.testingSents, network)
            evaluate(corpus)


def generateConf():
    kiperConf = configuration['kiperwasser']
    kiperConf['wordDim'] = int(generateValue([50, 250], True))
    kiperConf['posDim'] = int(generateValue([15, 75], True))
    kiperConf['denseActivation'] = 'tanh'  # str(generateValue(['tanh', 'relu'], False))
    kiperConf['optimizer'] = 'adagrad'  # str(generateValue(['adam', 'adagrad'], False))
    kiperConf['lr'] = 0.07  # round(generateValue([.01, .09], True), 3)
    # round(generateValue([.08, .3], True), 3) if kiperConf['optimizer'] == 'adam' else \
    #    round(generateValue([.001, .009], True), 3)
    kiperConf['dense1'] = int(generateValue([10, 150], True))
    kiperConf['lstmDropout'] = round(generateValue([.1, .4], True), 2)
    kiperConf['lstmUnitNum'] = int(generateValue([20, 150], True))
    kiperConf['lstmLayerNum'] = generateValue([1, 2], False)

    configuration['model']['embedding']['compactVocab'] = generateValue([True, False], False)
    configuration['model']['embedding']['lemma'] = generateValue([True, False], False)

    # sys.stdout.write('KIPER CONF: word = {0} - pos {1} - lstmLayers {2} - lstmUnits {3} - lstmDrop {4} ' \
    #                  '- denseUnits {5} - dnenseActiv {6} - optim {7} lr {8} compactVocab {9} Lemma {10} \n'.
    #                  format(kiperConf['wordDim'],
    #                         kiperConf['posDim'],
    #                         kiperConf['lstmLayerNum'],
    #                         kiperConf['lstmUnitNum'],
    #                         kiperConf['lstmDropout'],
    #                         kiperConf['dense1'],
    #                         kiperConf['denseActivation'],
    #                         kiperConf['optimizer'],
    #                         kiperConf['lr'],
    #                         configuration['model']['embedding']['compactVocab'],
    #                         configuration['model']['embedding']['lemma']
    #                         ))


def createRSGGrid(filename='kiperGrid.p'):
    conf = dict()
    for i in range(500):
        generateConf()
        conf[i] = [False, dict(configuration['kiperwasser']), dict(configuration['model']['embedding'])]
    pickle.dump(conf, open(os.path.join(configuration["path"]["projectPath"], 'ressources', filename), "wb"))
    return conf


def getRSGrid(filename='kiperGrid.p'):
    randomSearchGridPath = os.path.join(configuration["path"]["projectPath"], 'ressources', filename)
    return pickle.load(open(randomSearchGridPath, "rb"))


def generateValue(plage, continousPlage=False, uniform=False):
    if continousPlage:
        if uniform:
            return random.uniform(plage[0], plage[-1])
        else:
            return pow(2, random.uniform(math.log(plage[0], 2), math.log(plage[-1], 2)))
    else:
        return plage[random.randint(0, len(plage) - 1)]


def getActiveConfs():
    confs = getRSGrid()
    activeCons = 0
    for i in range(len(confs)):
        if confs[i][0]:
            print confs[i][1]
            activeCons += 1
    print activeCons


def runRSGThread(langs, xpNum=30, compact=False, filename='kiperGrid.p'):
    for i in range(xpNum):
        confs = getRSGrid(filename=filename)
        while True:
            confIdx = random.randint(1, len(confs)) - 1
            isRead = confs[confIdx][0]
            if not isRead:
                break
        confs[confIdx][0] = True
        pickle.dump(confs, open(os.path.join(configuration["path"]["projectPath"], 'ressources', "kiperGrid.p"), "wb"))
        configuration['kiperwasser'].update(confs[confIdx][1])
        configuration['model']['embedding'].update(confs[confIdx][2])
        print '# Kiper conf: ' + str(configuration['kiperwasser']) + str(
            configuration['model']['embedding']['lemma']) + ',' + str(
            configuration['model']['embedding']['compactVocab'])
        xp(langs, xpNum=1, compact=compact)


def exploreLR(langs, compact=False):
    lr = .0
    while True:
        lr += .01
        print '# Learning Rate : {0}'.format(lr)
        configuration['kiperwasser']['lr'] = lr
        xp(langs, xpNum=1, compact=compact)
        if lr >= .2:
            break


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    random.seed(0)
    torch.manual_seed(0)

    configuration['kiperwasser'].update({
        'focusedElemNum': 8,
        'wordDim': 100,
        'posDim': 30,
        'denseActivation': 'tanh',
        'dense1': 30,
        'dense2': 0,
        'denseDropout': False,
        'optimizer': 'adagrad',
        'lr': 0.07,
        'epochs': 15,
        'batch': 1,
        'lstmDropout': .2,
        'lstmLayerNum': 1,
        'lstmUnitNum': 60,
        'verbose': True,
        'file': 'kiperwasser.p'
    })
    configuration["sampling"]["importantSentences"] = True

    configuration["evaluation"]["fixedSize"] = True
    configuration['dataset']['sharedtask2'] = True
    configuration['xp']['kiperwasser'] = True
    # configuration['model']['embedding']['compactVocab'] = True
    # configuration['model']['embedding']['lemma'] = True
    configuration['kiperwasser']['earlyStop'] = True
    # for e in [5, 10, 15, 25, 35]:
    #     configuration['kiperwasser']['epochs'] = e
    #     xp(['FR'], xpNum=1, compact=False)
    # exploreLR(['BG'], True)
    # createRSGGrid()
    # getActiveConfs()
    configuration['kiperwasser']['epochs'] = 20
    xp(['BG'], xpNum=1, compact=False)
    # runRSGThread(['BG', 'PT', 'TR'])
