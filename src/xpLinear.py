import logging
import sys

import reports
from config import configuration, resetFRStandardFeatures
from identification import identify

allSharedtask1Lang = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT', \
                      'LT', 'MT', 'PL', 'PT', 'RO', 'SV', 'TR']

allSharedtask2Lang = ['BG', 'DE', 'EL', 'EN', 'ES', 'EU', 'FA', 'FR', 'HE', 'HI', \
                      'HR', 'HU', 'IT', 'LT', 'PL', 'PT', 'RO', 'SL', 'TR']


def exploreFeatures():
    evalConfig = configuration["evaluation"]
    evalConfig["debug"] = False
    evalConfig["train"] = True

    exploreLexicalFeatures()
    exploreNonLexicalFeatures()


def exploreNonLexicalFeatures(langs=['FR']):
    featConf = configuration["features"]

    resetFRStandardFeatures()

    reports.createHeader(' Title: Standard settings:  A B C E I J K L')
    identify(langs)

    reports.createHeader(' Title: Without syntax: A C E I J K L')
    featConf["syntax"]["active"] = False
    identify(langs)
    featConf["syntax"]["active"] = True

    reports.createHeader(' Title: Without BiGram: A B E I J K L')
    featConf["bigram"]["s0b2"] = False
    identify(langs)
    featConf["bigram"]["s0b2"] = True

    reports.createHeader(' Title: Without S0B2Bigram: A B C I J K L')
    featConf["bigram"]["s0b2"] = False
    identify(langs)
    featConf["bigram"]["s0b2"] = True

    reports.createHeader(' Title: Without S0B0Distance: A B C E J K L')
    featConf["distance"]["s0b0"] = False
    identify(langs)
    featConf["distance"]["s0b0"] = True

    reports.createHeader(' Title: Without S0S1Distance: A B C E I K L')
    featConf["distance"]["s0s1"] = False
    identify(langs)
    featConf["distance"]["s0s1"] = True

    reports.createHeader(' Title: without B1:  A B C E I J L')
    featConf["unigram"]["b1"] = False
    identify(langs)
    featConf["unigram"]["b1"] = True

    reports.createHeader(' Title: without lexicon:  A B C E I J K')
    featConf["dictionary"]["active"] = False
    identify(langs)


def exploreLexicalFeatures(langs=['FR']):
    evalConfig = configuration["evaluation"]
    evalConfig["debug"] = False
    evalConfig["train"] = True
    resetNonLexicalFeatures()
    # token
    explore(langs, True, False, False)
    # pos
    explore(langs, False, False, True)
    # lemma
    explore(langs, False, True, False)
    # pos + token
    explore(langs, True, False, True)
    # pos + lemma
    explore(langs, False, True, True)
    # token + lemma
    explore(langs, True, True, False)
    # token + lemma + POS
    explore(langs, True, True, True)


def exploreAbstractSyntax(langs=['FR']):
    featConf = configuration["features"]
    featConf["unigram"]["pos"] = True
    featConf["unigram"]["token"] = False
    featConf["unigram"]["lemma"] = True
    featConf["bigram"]["active"] = True
    identify(langs)
    featConf["syntax"]["abstract"] = True
    identify(langs)


def explore(langs=['FR'], token=False, lemma=False, pos=False):
    featConf = configuration["features"]
    featConf["unigram"]["pos"] = pos
    featConf["unigram"]["token"] = token
    featConf["unigram"]["lemma"] = lemma
    featConf["bigram"]["active"] = True
    featConf["dictionary"]["mwt"] = False
    title = 'token ' if token else ''
    title += 'lemma ' if lemma else ''
    title += 'pos ' if pos else ''
    reports.createHeader(' Title: uni + bi: {0}'.format(title))
    identify(langs)
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


def discoverSampling(langs=['FR'], importantSentences=True, overSampling=True):
    resetNonLexicalFeatures()
    configuration["model"]["train"]["sampling"]["importantSentences"] = importantSentences
    configuration["model"]["train"]["sampling"]["overSampling"] = overSampling
    explore(langs, lemma=True, pos=True, token=False)


def exploreSharedtask2(langs=['FR']):
    configuration["evaluation"]["sharedtask2"] = True
    resetNonLexicalFeatures()
    explore(langs, lemma=True, pos=True, token=False)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    configuration["model"]["train"]["linear"] = True

    configuration["evaluation"]["debug"] = False
    configuration["evaluation"]["train"] = True

    exploreSharedtask2()
