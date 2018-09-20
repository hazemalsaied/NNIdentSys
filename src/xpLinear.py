import logging
import os
import pickle
import random
import sys

import reports
from config import Dataset, Evaluation, XpMode
from config import configuration, resetFRStandardFeatures, resetStandardFeatures
from identification import setDataSet, setTrainAndTest, setXPMode
from identification import xp


def exploreFeatures():
    exploreLexicalFeatures()
    exploreNonLexicalFeatures()


def exploreNonLexicalFeatures():
    featConf = configuration['features']

    resetStandardFeatures()

    reports.createHeader(' Title: Standard settings:  A B C E I J K L')
    configuration['features'].update({

    })
    xp(langs, xpNum=1)

    reports.createHeader(' Title: Without syntax: A C E I J K L')
    featConf['syntax']['active'] = False
    xp(langs, xpNum=1)
    featConf['syntax']['active'] = True

    reports.createHeader(' Title: Without BiGram: A B E I J K L')
    featConf['bigram']['s0b2'] = False
    xp(langs, xpNum=1)
    featConf['bigram']['s0b2'] = True

    reports.createHeader(' Title: Without S0B2Bigram: A B C I J K L')
    featConf['bigram']['s0b2'] = False
    xp(langs, xpNum=1)
    featConf['bigram']['s0b2'] = True

    reports.createHeader(' Title: Without S0B0Distance: A B C E J K L')
    featConf['distance']['s0b0'] = False
    xp(langs, xpNum=1)
    featConf['distance']['s0b0'] = True

    reports.createHeader(' Title: Without S0S1Distance: A B C E I K L')
    featConf['distance']['s0s1'] = False
    xp(langs, xpNum=1)
    featConf['distance']['s0s1'] = True

    reports.createHeader(' Title: without B1:  A B C E I J L')
    featConf['unigram']['b1'] = False
    xp(langs, xpNum=1)
    featConf['unigram']['b1'] = True

    reports.createHeader(' Title: without lexicon:  A B C E I J K')
    featConf['dictionary']['active'] = False
    xp(langs, xpNum=1)


def exploreLexicalFeatures():
    # token
    explore(True, False, False)
    # pos
    explore(False, False, True)
    # lemma
    explore(False, True, False)
    # pos + token
    explore(True, False, True)
    # pos + lemma
    explore(False, True, True)
    # token + lemma
    explore(True, True, False)
    # token + lemma + POS
    explore(True, True, True)


def exploreAbstractSyntax():
    featConf = configuration['features']
    featConf['unigram']['pos'] = True
    featConf['unigram']['token'] = False
    featConf['unigram']['lemma'] = True
    featConf['bigram']['active'] = True
    xp(langs, xpNum=1)
    featConf['syntax']['abstract'] = True
    xp(langs, xpNum=1)


def explore(token=False, lemma=False, pos=False,
            ):
    resetStandardFeatures()
    featConf = configuration['features']
    featConf['unigram']['pos'] = pos
    featConf['unigram']['token'] = token
    featConf['unigram']['lemma'] = lemma
    featConf['bigram']['active'] = True
    title = 'token ' if token else ''
    title += 'lemma ' if lemma else ''
    title += 'pos ' if pos else ''
    reports.createHeader(' Title: uni + bi: {0}'.format(title))
    xp(langs, xpNum=1)


def resetNonLexicalFeatures(value=False):
    featConf = configuration['features']
    featConf['syntax']['active'] = value
    featConf['syntax']['abstract'] = value
    featConf['bigram']['s0b2'] = value
    featConf['bigram']['active'] = value
    featConf['trigram'] = value
    featConf['distance']['s0b0'] = value
    featConf['distance']['s0s1'] = value
    featConf['stackLength'] = value
    featConf['unigram']['b1'] = value
    featConf['dictionary']['active'] = value
    featConf['history']['1'] = value
    featConf['history']['2'] = value
    featConf['history']['3'] = value
    featConf['unigram']['pos'] = value
    featConf['unigram']['lemma'] = value


def discoverSampling(importantSentences=True, overSampling=True):
    resetNonLexicalFeatures()
    configuration['sampling']['importantSentences'] = importantSentences
    configuration['sampling']['overSampling'] = overSampling
    explore(lemma=True, pos=True, token=False)


def exploreSharedtask2():
    configuration['dataset']['sharedtask2'] = True
    resetNonLexicalFeatures()
    explore(lemma=True, pos=True, token=False)


def exploreFTB():
    configuration['dataset']['FTB'] = True
    resetFRStandardFeatures()
    # resetNonLexicalFeatures()
    # explore(lemma=True, pos=True, token=False)
    xp(langs, xpNum=1)


def runRSG(fileName='linear.p', xpNumByThread=15):
    for i in range(xpNumByThread):
        exps = getGrid(fileName)
        while True:
            xpIdx = random.randint(1, len(exps)) - 1
            exp = exps.keys()[xpIdx]
            if not exps[exp][0]:
                break
        exps[exp][0] = True
        pickle.dump(exps, open(os.path.join(configuration['path']['projectPath'], 'ressources', fileName), 'wb'))
        configuration['features'].update(exps[exp][1])
        sys.stdout.write('\n# Features Conf :{0}\n'.format(str(configuration['features'])))
        xp(langs, xpNum=1)


def getActivatedConf(fileName='linear.p'):
    exps = getGrid(fileName)
    for i in range(len(exps)):
        if exps[i][0]:
            print exps[i][1]


def createRSGrid(confNum=1000, fileName='linear.p'):
    confs, generatedConfs = dict(), set()
    idx = 0
    for i in range(confNum):
        conf = {
            'active': True,
            'unigram': {
                'lemma': True,
                'token': generateValue(.2),
                'pos': True,
                'suffix': False,
                'b1': generateValue(.4)
            },
            'bigram': {
                'active': True,
                's0b2': generateValue(.5)
            },
            'trigram': generateValue(.5),
            'syntax': {
                'active': False,
                'abstract': False,
                'lexicalised': False,
                'bufferElements': 5
            },
            'dictionary': {
                'active': generateValue(.5),
                'mwt': True,
                's0TokenIsMWEToken': generateValue(.5),
                's0TokensAreMWE': False
            },
            'history': {
                '1': generateValue(.5),
                '2': generateValue(.5),
                '3': generateValue(.5)
            },
            'stackLength': generateValue(.5),
            'distance': {
                's0s1': generateValue(.5),
                's0b0': generateValue(.5)
            }
        }
        confKey = getKey(conf)
        if confKey not in generatedConfs:
            print confKey
            confs[idx] = [False, conf]
            generatedConfs.add(confKey)
            idx += 1
    pickle.dump(confs, open(os.path.join(configuration['path']['projectPath'],
                                         'ressources', fileName), 'wb'))


def getKey(conf):
    s = 's0s1' if conf['distance']['s0s1'] else ''
    s += 's0b0' if conf['distance']['s0b0'] else ''
    s += 'stackLength' if conf['stackLength'] else ''
    s += 'H1' if conf['history']['1'] else ''
    s += 'H2' if conf['history']['2'] else ''
    s += 'H3' if conf['history']['3'] else ''
    s += 'dictionary' if conf['dictionary']['active'] else ''
    s += 's0TokenIsMWEToken' if conf['dictionary']['s0TokenIsMWEToken'] else ''
    s += 'trigram' if conf['trigram'] else ''
    s += 's0b2' if conf['bigram']['s0b2'] else ''
    s += 'token' if conf['unigram']['token'] else ''
    s += 'b1' if conf['unigram']['b1'] else ''
    return s


def getGrid(fileName='linear.p'):
    randomSearchGridPath = os.path.join(configuration['path']['projectPath'], 'ressources', fileName)
    return pickle.load(open(randomSearchGridPath, 'rb'))


def generateValue(favorisationTaux=0.5):
    alpha = random.uniform(0, 1)
    if alpha < favorisationTaux:
        return True
    return False


def setOptimalRSGFeatures():
    conf = {
        'active': True,
        'unigram': {
            'lemma': True,
            'token': False,
            'pos': True,
            'suffix': False,
            'b1': False
        },
        'bigram': {
            'active': True,
            's0b2': False
        },
        'trigram': True,
        'syntax': {
            'active': False,
            'abstract': False,
            'lexicalised': False,
            'bufferElements': 5
        },
        'dictionary': {
            'active': False,
            'mwt': True,
            's0TokenIsMWEToken': False,
            's0TokensAreMWE': False
        },
        'history': {
            '1': True,
            '2': True,
            '3': True
        },
        'stackLength': False,
        'distance': {
            's0s1': True,
            's0b0': True
        }
    }
    configuration['features'].update(conf)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    setXPMode(XpMode.linear)

    langs = ['FR']

    setDataSet(Dataset.sharedtask2)

    from xpNonCompo import allSharedtask2Lang

    setOptimalRSGFeatures()
    setTrainAndTest(Evaluation.corpus)
    xp(allSharedtask2Lang, xpNum=1)

    setTrainAndTest(Evaluation.trainVsTest)
    xp(allSharedtask2Lang, xpNum=1)

    setTrainAndTest(Evaluation.trainVsDev)
    xp(allSharedtask2Lang, xpNum=1)

    # langs = ['BG', 'PT', 'TR']
    # configuration['linear']['svm'] = False
    # createRSGrid()
    # runRSG()
    # getActivatedConf()
