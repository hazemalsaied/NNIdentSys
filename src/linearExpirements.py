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

    reports.createHeader(' Title: Standard settings:  A B C E I J K L')
    identifyV2()

    reports.createHeader(' Title: Without syntax: A C E I J K L')
    featConf["syntax"]["active"] = False
    identifyV2()
    featConf["syntax"]["active"] = True

    reports.createHeader(' Title: Without BiGram: A B E I J K L')
    featConf["bigram"]["s0b2"] = False
    identifyV2()
    featConf["bigram"]["s0b2"] = True

    reports.createHeader(' Title: Without S0B2Bigram: A B C I J K L')
    featConf["bigram"]["s0b2"] = False
    identifyV2()
    featConf["bigram"]["s0b2"] = True

    reports.createHeader(' Title: Without S0B0Distance: A B C E J K L')
    featConf["distance"]["s0b0"] = False
    identifyV2()
    featConf["distance"]["s0b0"] = True

    reports.createHeader(' Title: Without S0S1Distance: A B C E I K L')
    featConf["distance"]["s0s1"] = False
    identifyV2()
    featConf["distance"]["s0s1"] = True

    reports.createHeader(' Title: without B1:  A B C E I J L')
    featConf["unigram"]["b1"] = False
    identifyV2()
    featConf["unigram"]["b1"] = True

    reports.createHeader(' Title: without lexicon:  A B C E I J K')
    featConf["dictionary"]["active"] = False
    identifyV2()


def exploreLexicalFeatures(langs):
    evalConfig = configuration["evaluation"]
    evalConfig["debug"] = False
    evalConfig["train"] = True
    resetNonLexicalFeatures()
    # token
    #explore(True, False, False)
    # pos
    #explore(False, False, True)
    # lemma
    #explore(False, True, False)
    # pos + token
    #explore(True, False, True)
    # pos + lemma
    explore(langs, False, True, True)
    # token + lemma
    #explore(True, True, False)
    # token + lemma + POS
    #explore(True, True, True)

def exploreAbstractSyntax():
    featConf = configuration["features"]
    featConf["unigram"]["pos"] = True
    featConf["unigram"]["token"] = False
    featConf["unigram"]["lemma"] = True
    featConf["bigram"]["active"] = True
    identifyV2()
    featConf["syntax"]["abstract"] = True
    identifyV2()


def explore(langs, token=False, lemma=False, pos=False):
    featConf = configuration["features"]
    featConf["unigram"]["pos"] = pos
    featConf["unigram"]["token"] = token
    featConf["unigram"]["lemma"] = lemma
    title = 'token ' if token else ''
    title += 'lemma ' if lemma else ''
    title += 'pos ' if pos else ''

    #reports.createHeader(' Title: uni: {0}'.format(title))
    #identifyV2()
    featConf["bigram"]["active"] = True
    reports.createHeader(' Title: uni + bi: {0}'.format(title))
    identifyV2(langs)
    #featConf["trigram"] = True
    #reports.createHeader(' Title: uni + bi + tri: {0}'.format(title))
    #identifyV2()
    #featConf["bigram"]["active"] = False
    #featConf["trigram"] = False
    #featConf["bigram"]["active"] = False


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

    evalConfig = configuration["evaluation"]
    evalConfig["debug"] = False
    evalConfig["train"] = True

    exploreAbstractSyntax()
    exploreLexicalFeatures(['BG','CS', 'DE', 'EL','ES','FA','FR','HE','HU','IT','LT','MT','PL','PT','RO','SV','TR'] )
