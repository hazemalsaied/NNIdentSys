import logging
import sys

import reports
from config import configuration, resetFRStandardFeatures
from identification import identify

allSharedtask1Lang = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT',
                      'LT', 'MT', 'PL', 'PT', 'RO', 'SV', 'TR']

allSharedtask2Lang = ['BG', 'DE', 'EL', 'EN', 'ES', 'EU', 'FA', 'FR', 'HE', 'HI',
                      'HR', 'HU', 'IT', 'LT', 'PL', 'PT', 'RO', 'SL', 'TR']

langs = ['FR']


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
    for lang in langs:
        identify(lang)

    reports.createHeader(' Title: Without syntax: A C E I J K L')
    featConf["syntax"]["active"] = False
    for lang in langs:
        identify(lang)
    featConf["syntax"]["active"] = True

    reports.createHeader(' Title: Without BiGram: A B E I J K L')
    featConf["bigram"]["s0b2"] = False
    for lang in langs:
        identify(lang)
    featConf["bigram"]["s0b2"] = True

    reports.createHeader(' Title: Without S0B2Bigram: A B C I J K L')
    featConf["bigram"]["s0b2"] = False
    for lang in langs:
        identify(lang)
    featConf["bigram"]["s0b2"] = True

    reports.createHeader(' Title: Without S0B0Distance: A B C E J K L')
    featConf["distance"]["s0b0"] = False
    for lang in langs:
        identify(lang)
    featConf["distance"]["s0b0"] = True

    reports.createHeader(' Title: Without S0S1Distance: A B C E I K L')
    featConf["distance"]["s0s1"] = False
    for lang in langs:
        identify(lang)
    featConf["distance"]["s0s1"] = True

    reports.createHeader(' Title: without B1:  A B C E I J L')
    featConf["unigram"]["b1"] = False
    for lang in langs:
        identify(lang)
    featConf["unigram"]["b1"] = True

    reports.createHeader(' Title: without lexicon:  A B C E I J K')
    featConf["dictionary"]["active"] = False
    identify(langs)


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


def exploreAbstractSyntax():
    featConf = configuration["features"]
    featConf["unigram"]["pos"] = True
    featConf["unigram"]["token"] = False
    featConf["unigram"]["lemma"] = True
    featConf["bigram"]["active"] = True
    for lang in langs:
        identify(lang)
    featConf["syntax"]["abstract"] = True
    for lang in langs:
        identify(lang)


def explore(token=False, lemma=False, pos=False, mwt=False):
    featConf = configuration["features"]
    featConf["unigram"]["pos"] = pos
    featConf["unigram"]["token"] = token
    featConf["unigram"]["lemma"] = lemma
    featConf["bigram"]["active"] = True
    featConf["dictionary"]["mwt"] = mwt
    title = 'token ' if token else ''
    title += 'lemma ' if lemma else ''
    title += 'pos ' if pos else ''
    reports.createHeader(' Title: uni + bi: {0}'.format(title))
    for lang in langs:
        identify(lang)


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


def discoverSampling(importantSentences=True, overSampling=True):
    resetNonLexicalFeatures()
    configuration["sampling"]["importantSentences"] = importantSentences
    configuration["sampling"]["overSampling"] = overSampling
    explore(lemma=True, pos=True, token=False)


def exploreSharedtask2():
    configuration["dataset"]["sharedtask2"] = True
    resetNonLexicalFeatures()
    explore(lemma=True, pos=True, token=False)


def exploreFTB():
    configuration["dataset"]["FTB"] = True
    resetFRStandardFeatures()
    # resetNonLexicalFeatures()
    # explore(lemma=True, pos=True, token=False)
    for lang in langs:
        identify(lang)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    langs = ['BG', 'PT', 'TR']
    configuration["xp"]["linear"] = True
    configuration["evaluation"]["debug"] = False
    configuration["evaluation"]["fixedSize"] = True
    configuration["dataset"]["sharedtask2"] = True
    exploreSharedtask2()
    # configuration["evaluation"]["corpus"] = True
    # exploreFTB()
