import logging
import sys

from config import configuration
from reports import createHeader
from xpTools import xp, Dataset, Evaluation, XpMode


def resetFRStandardFeatures():
    conf = {
        'lemma': True,
        'token': True,
        'pos': True,
        'suffix': False,
        'b1': True,
        'bigram': True,
        's0b2': True,
        'trigram': False,
        'syntax': True,
        'syntaxAbstract': True,
        'dictionary': True,
        's0TokenIsMWEToken': False,
        's0TokensAreMWE': False,
        'history1': False,
        'history2': False,
        'history3': False,
        'stackLength': False,
        'distanceS0s1': True,
        'distanceS0b0': True
    }
    configuration['features'].update(conf)


def resetStandardFeatures(v=False):
    configuration['features'].update({
        'lemma': v,
        'token': v,
        'pos': v,
        'suffix': v,
        'b1': v,
        'bigram': v,
        's0b2': v,
        'trigram': v,
        'syntax': v,
        'syntaxAbstract': v,
        'dictionary': v,
        's0TokenIsMWEToken': v,
        's0TokensAreMWE': v,
        'history1': v,
        'history2': v,
        'history3': v,
        'stackLength': v,
        'distanceS0s1': v,
        'distanceS0b0': v
    })


def exploreFeatures(langs, dataset, xpMode, division):
    exploreLexicalFeatures(langs, dataset, xpMode, division)
    exploreNonLexicalFeatures(langs, dataset, xpMode, division)


def exploreNonLexicalFeatures(langs, dataset, xpMode, division):
    featConf = configuration['features']

    resetStandardFeatures()

    createHeader(' Title: Standard settings:  A B C E I J K L')
    configuration['features'].update({

    })
    xp(langs, dataset, xpMode, division)

    createHeader(' Title: Without syntax: A C E I J K L')
    featConf['syntax'] = False
    xp(langs, dataset, xpMode, division)
    featConf['syntax'] = True

    createHeader(' Title: Without BiGram: A B E I J K L')
    featConf['s0b2'] = False
    xp(langs, dataset, xpMode, division)
    featConf['s0b2'] = True

    createHeader(' Title: Without S0B2Bigram: A B C I J K L')
    featConf['s0b2'] = False
    xp(langs, dataset, xpMode, division)
    featConf['s0b2'] = True

    createHeader(' Title: Without S0B0Distance: A B C E J K L')
    featConf['distance']['s0b0'] = False
    xp(langs, dataset, xpMode, division)
    featConf['distance']['s0b0'] = True

    createHeader(' Title: Without S0S1Distance: A B C E I K L')
    featConf['distanceS0s1'] = False
    xp(langs, dataset, xpMode, division)
    featConf['distanceS0s1'] = True

    createHeader(' Title: without B1:  A B C E I J L')
    featConf['b1'] = False
    xp(langs, dataset, xpMode, division)
    featConf['b1'] = True

    createHeader(' Title: without lexicon:  A B C E I J K')
    featConf['dictionary'] = False
    xp(langs, dataset, xpMode, division)


def exploreLexicalFeatures(langs, dataset, xpMode, division):
    # token
    explore(langs, dataset, xpMode, division, True, False, False)
    # pos
    explore(langs, dataset, xpMode, division, False, False, True)
    # lemma
    explore(langs, dataset, xpMode, division, False, True, False)
    # pos + token
    explore(langs, dataset, xpMode, division, True, False, True)
    # pos + lemma
    explore(langs, dataset, xpMode, division, False, True, True)
    # token + lemma
    explore(langs, dataset, xpMode, division, True, True, False)
    # token + lemma + POS
    explore(langs, dataset, xpMode, division, True, True, True)


def exploreAbstractSyntax(langs, dataset, xpMode, division):
    featConf = configuration['features']
    featConf['pos'] = True
    featConf['token'] = False
    featConf['lemma'] = True
    featConf['bigram'] = True
    xp(langs, dataset, xpMode, division)
    featConf['syntaxAbstract'] = True
    xp(langs, dataset, xpMode, division)


def explore(langs, dataset, xpMode, division, token=False, lemma=False, pos=False):
    resetStandardFeatures()
    featConf = configuration['features']
    configuration['features']['pos'] = pos
    configuration['features']['token'] = token
    configuration['features']['lemma'] = lemma
    configuration['features']['bigram'] = True
    title = 'token ' if token else ''
    title += 'lemma ' if lemma else ''
    title += 'pos ' if pos else ''
    createHeader(' Title: uni + bi: {0}'.format(title))
    xp(langs, dataset, xpMode, division)


def resetNonLexicalFeatures(value=False):
    conf = {
        'lemma': True,
        'pos': True,
        'token': value,
        'suffix': value,
        'b1': value,
        'bigram': value,
        's0b2': value,
        'trigram': value,
        'syntax': value,
        'syntaxAbstract': value,
        'dictionary': value,
        's0TokenIsMWEToken': value,
        's0TokensAreMWE': value,
        'history1': value,
        'history2': value,
        'history3': value,
        'stackLength': value,
        'distanceS0s1': value,
        'distanceS0b0': value
    }
    configuration['features'].update(conf)


def discoverSampling(langs, dataset, xpMode, division, importantSentences=True, overSampling=True):
    resetNonLexicalFeatures()
    configuration['sampling']['importantSentences'] = importantSentences
    configuration['sampling']['overSampling'] = overSampling
    explore(langs, dataset, xpMode, division, lemma=True, pos=True, token=False)


def setMinimalFeatures():
    conf = {
        'lemma': True,
        'token': False,
        'pos': True,
        'suffix': False,
        'b1': False,
        'bigram': True,
        's0b2': False,
        'trigram': False,
        'syntax': False,
        'syntaxAbstract': False,
        'dictionary': False,
        's0TokenIsMWEToken': False,
        's0TokensAreMWE': False,
        'history1': False,
        'history2': False,
        'history3': False,
        'stackLength': False,
        'distanceS0s1': False,
        'distanceS0b0': False
    }
    configuration['features'].update(conf)


def setOptimalRSGFeaturesByLogReg():
    conf = {
        'lemma': True,
        'token': False,
        'pos': True,
        'suffix': False,
        'b1': False,
        'bigram': True,
        's0b2': False,
        'trigram': True,
        'syntax': False,
        'syntaxAbstract': False,
        'dictionary': False,
        's0TokenIsMWEToken': False,
        's0TokensAreMWE': False,
        'history1': True,
        'history2': True,
        'history3': True,
        'stackLength': False,
        'distanceS0s1': True,
        'distanceS0b0': True
    }
    configuration['features'].update(conf)


def setOptimalRSGFeaturesBySVM():
    conf = {
        'lemma': True,
        'token': True,
        'pos': True,
        'suffix': False,
        'b1': True,
        'bigram': True,
        's0b2': True,
        'trigram': True,
        'syntax': False,
        'syntaxAbstract': False,
        'dictionary': True,
        's0TokenIsMWEToken': True,
        's0TokensAreMWE': True,
        'history1': True,
        'history2': False,
        'history3': True,
        'stackLength': False,
        'distanceS0s1': True,
        'distanceS0b0': True
    }
    configuration['features'].update(conf)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    mode = XpMode.linear
    dset = Dataset.sharedtask2
    evaluation = Evaluation.fixedSize
    # setMinimalFeatures()
    resetFRStandardFeatures()
    xp(['FR'], Dataset.dimsum, None, Evaluation.corpus)

    # from xpTools import allSharedtask2Lang
    #
    # # setMinimalFeatures()
    # setOptimalRSGFeaturesBySVM()
    # xp(allSharedtask2Lang, dset, mode, Evaluation.corpus)
    # xp(allSharedtask2Lang, dset, mode, Evaluation.trainVsTest)
    # xp(allSharedtask2Lang, dset, mode, Evaluation.trainVsDev)
    #
    # xp(['FR'], Dataset.ftb, mode, Evaluation.corpus)
    # xp(['FR'], Dataset.ftb, mode, Evaluation.trainVsTest)
    # xp(['FR'], Dataset.ftb, mode, Evaluation.trainVsDev)
    # xp(['EN'], Dataset.dimsum, mode, Evaluation.corpus)
    # from xpTools import pilotLangs
    # from rsg import runRSG
    # createRSG('linear.p', XpMode.linear)
    # runRSG(pilotLangs, dset, mode, evaluation, 'linear.p', xpNumByThread=150)  # Evaluation.fixedSize
