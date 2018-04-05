import logging
import sys

import reports
from config import configuration, resetFRStandardFeatures
from identification import identifyV2


def exploreFeatures():
    evalConfig = configuration["evaluation"]
    evalConfig["debug"] = False
    evalConfig["train"] = True

    exploreLexicalFeatures()
    exploreNonLexicalFeatures()


def exploreNonLexicalFeatures():
    featConf = configuration["features"]

    resetFRStandardFeatures()

    reports.createHeader('Standard settings:  A B C E I J K L')
    identifyV2()

    reports.createHeader('Without syntax: A C E I J K L')
    featConf["syntax"]["active"] = False
    identifyV2()
    featConf["syntax"]["active"] = True

    reports.createHeader('Without BiGram: A B E I J K L')
    featConf["bigram"]["s0b2"] = False
    identifyV2()
    featConf["bigram"]["s0b2"] = True

    reports.createHeader('Without S0B2Bigram: A B C I J K L')
    featConf["bigram"]["s0b2"] = False
    identifyV2()
    featConf["bigram"]["s0b2"] = True

    reports.createHeader('Without S0B0Distance: A B C E J K L')
    featConf["distance"]["s0b0"] = False
    identifyV2()
    featConf["distance"]["s0b0"] = True

    reports.createHeader('Without S0S1Distance: A B C E I K L')
    featConf["distance"]["s0s1"] = False
    identifyV2()
    featConf["distance"]["s0s1"] = True

    reports.createHeader('without B1:  A B C E I J L')
    featConf["unigram"]["b1"] = False
    identifyV2()
    featConf["unigram"]["b1"] = True

    reports.createHeader('without lexicon:  A B C E I J K')
    featConf["dictionary"]["active"] = False
    identifyV2()
    featConf["dictionary"]["active"] = False

    featConf["syntax"]["active"] = False
    featConf["bigram"]["active"] = False
    featConf["bigram"]["s0b2"] = False
    featConf["dictionary"]["active"] = False
    featConf["unigram"]["b1"] = False

    featConf["distance"]["s0s1"] = False
    featConf["distance"]["s0b0"] = False
    reports.createHeader('Unigram only:  A ')
    identifyV2()

    featConf["unigram"]["pos"] = False
    featConf["unigram"]["lemma"] = False
    reports.createHeader('token  only')
    identifyV2()

    reports.createHeader('lemma + token ')
    featConf["unigram"]["lemma"] = True
    identifyV2()

    reports.createHeader('Pos + token')
    featConf["unigram"]["lemma"] = False
    featConf["unigram"]["pos"] = True
    identifyV2()


def exploreLexicalFeatures():
    evalConfig = configuration["evaluation"]
    evalConfig["debug"] = False
    evalConfig["train"] = True
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


def explore(token=False, lemma=False, pos=False):
    featConf = configuration["features"]
    featConf["unigram"]["pos"] = pos
    featConf["unigram"]["token"] = token
    featConf["unigram"]["lemma"] = lemma
    title = 'token ' if token else ''
    title += 'lemma ' if lemma else ''
    title += 'pos ' if pos else ''

    reports.createHeader('uni: {0}'.format(title))
    identifyV2()
    featConf["bigram"]["active"] = True
    reports.createHeader('uni + bi: {0}'.format(title))
    identifyV2()
    featConf["trigram"] = True
    reports.createHeader('uni + bi + tri: {0}'.format(title))
    identifyV2()
    featConf["bigram"]["active"] = False
    featConf["trigram"] = False
    featConf["bigram"]["active"] = False


def resetNonLexicalFeatures(desactivate=False):
    featConf = configuration["features"]

    featConf["syntax"]["active"] = desactivate
    featConf["syntax"]["abstract"] = desactivate
    featConf["bigram"]["s0b2"] = desactivate
    featConf["bigram"]["active"] = desactivate
    featConf["trigram"] = desactivate
    featConf["distance"]["s0b0"] = desactivate
    featConf["distance"]["s0s1"] = desactivate
    featConf["stackLength"] = desactivate
    featConf["unigram"]["b1"] = desactivate
    featConf["dictionary"]["active"] = desactivate
    featConf["history"]["1"] = desactivate
    featConf["history"]["2"] = desactivate
    featConf["history"]["3"] = desactivate
    featConf["unigram"]["pos"] = desactivate
    featConf["unigram"]["lemma"] = desactivate




if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    exploreFeatures()
