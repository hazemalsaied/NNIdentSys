import logging
import sys

import reports
from config import configuration, resetFRStandardFeatures
from identification import xp


def exploreFeatures():
    evalConfig = configuration['evaluation']
    evalConfig['dev'] = True

    exploreLexicalFeatures()
    exploreNonLexicalFeatures()


def exploreNonLexicalFeatures():
    featConf = configuration['features']

    resetFRStandardFeatures()

    reports.createHeader(' Title: Standard settings:  A B C E I J K L')
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
    evalConfig = configuration['evaluation']
    evalConfig['dev'] = True
    resetNonLexicalFeatures()
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


def explore(token=False, lemma=False, pos=False, mwt=False):
    featConf = configuration['features']
    featConf['unigram']['pos'] = pos
    featConf['unigram']['token'] = token
    featConf['unigram']['lemma'] = lemma
    featConf['bigram']['active'] = True
    featConf['dictionary']['mwt'] = mwt
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
    explore(lemma=True, pos=True, token=False, mwt=True)


def exploreFTB():
    configuration['dataset']['FTB'] = True
    resetFRStandardFeatures()
    # resetNonLexicalFeatures()
    # explore(lemma=True, pos=True, token=False)
    xp(langs, xpNum=1)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    configuration['xp']['linear'] = True

    from xpNonCompo import allSharedtask2Lang

    langs = allSharedtask2Lang  # ['FR']
    configuration['dataset']['sharedtask2'] = True
    configuration['linear']['svm'] = True

    configuration['evaluation']['corpus'] = True
    exploreSharedtask2()

    configuration['evaluation']['trainVsTest'] = True
    exploreSharedtask2()

    configuration['evaluation']['trainVsDev'] = True
    exploreSharedtask2()
