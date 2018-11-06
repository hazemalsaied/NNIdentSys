# from rsg import *
from xpTools import *


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
        'syntax': False,
        'syntaxAbstract': False,
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
        's0TokensAreMWE': False,
        'history1': True,
        'history2': False,
        'history3': True,
        'stackLength': False,
        'distanceS0s1': True,
        'distanceS0b0': True
    }
    configuration['features'].update(conf)


def setOptimalRSGFeaturesForDimsumSVM():
    conf = {
        'lemma': True,
        'token': False,
        'pos': True,
        'suffix': False,
        'b1': True,
        'bigram': True,
        's0b2': False,
        'trigram': True,
        'syntax': False,
        'syntaxAbstract': False,
        'dictionary': True,
        's0TokenIsMWEToken': False,
        's0TokensAreMWE': False,
        'history1': True,
        'history2': False,
        'history3': True,
        'stackLength': False,
        'distanceS0s1': False,
        'distanceS0b0': True
    }
    configuration['features'].update(conf)

def setOptimalRSGFeaturesForFtbSVM():
    conf = {
        'lemma': True,
        'token': True,
        'pos': True,
        'suffix': False,
        'b1': False,
        'bigram': True,
        's0b2': False,
        'trigram': True,
        'syntax': False,
        'syntaxAbstract': False,
        'dictionary': True,
        's0TokenIsMWEToken': False,
        's0TokensAreMWE': False,
        'history1': False,
        'history2': False,
        'history3': False,
        'stackLength': True,
        'distanceS0s1': True,
        'distanceS0b0': True
    }
    configuration['features'].update(conf)


def setOptimalRSGForMlpFTB():
    samling = configuration['sampling']
    samling['importantSentences'] = True
    samling['overSampling'] = True
    samling['sampleWeight'] = True
    samling['favorisationCoeff'] = 3
    samling['focused'] = True

    configuration['mlp']['optimizer'] = 'adagrad'
    configuration['mlp']['lr'] = 0.059

    configuration['mlp']['lemma'] = True
    configuration['mlp']['posEmb'] = 103
    configuration['mlp']['tokenEmb'] = 410
    configuration['mlp']['compactVocab'] = True

    configuration['mlp']['dense1'] = True
    configuration['mlp']['dense1UnitNumber'] = 167
    configuration['mlp']['dense1Dropout'] = 0.16

def setOptimalRSGForMlpDiMSUM():
    samling = configuration['sampling']
    samling['importantSentences'] = True
    samling['overSampling'] = True
    samling['sampleWeight'] = True
    samling['favorisationCoeff'] = 13
    samling['focused'] = True

    configuration['mlp']['optimizer'] = 'adagrad'
    configuration['mlp']['lr'] = 0.059

    configuration['mlp']['lemma'] = True
    configuration['mlp']['posEmb'] = 118
    configuration['mlp']['tokenEmb'] = 326
    configuration['mlp']['compactVocab'] = True

    configuration['mlp']['dense1'] = True
    configuration['mlp']['dense1UnitNumber'] = 172
    configuration['mlp']['dense1Dropout'] = 0.38

def setOptimalRSGForMLP():
    samling = configuration['sampling']
    samling['importantSentences'] = True
    samling['overSampling'] = True
    samling['sampleWeight'] = True
    samling['favorisationCoeff'] = 6
    samling['focused'] = True

    configuration['mlp']['optimizer'] = 'adagrad'
    configuration['mlp']['lr'] = 0.059

    configuration['mlp']['lemma'] = True
    configuration['mlp']['posEmb'] = 42
    configuration['mlp']['tokenEmb'] = 480
    configuration['mlp']['compactVocab'] = False

    configuration['mlp']['dense1'] = True
    configuration['mlp']['dense1UnitNumber'] = 58
    configuration['mlp']['dense1Dropout'] = 0.429


def setOptimalRSGForRNN():
    configuration['rnn']['gru'] = True
    configuration['rnn']['useDense'] = True
    configuration['mlp']['compactVocab'] = True
    configuration['mlp']['lemma'] = True

    configuration['sampling']['importantSentences'] = True
    configuration['sampling']['overSampling'] = True

    configuration['rnn']['wordDim'] = 410
    configuration['rnn']['posDim'] = 54
    configuration['rnn']['denseUnitNum'] = 51
    configuration['rnn']['denseDropout'] = 0.1
    configuration['rnn']['wordRnnUnitNum'] = 25
    configuration['rnn']['posRnnUnitNum'] = 15
    configuration['rnn']['rnnDropout'] = 0.1
    configuration['rnn']['batchSize'] = 16


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    # setOptimalRSGFeaturesBySVM()
    # xp(['FR'], Dataset.ftb, XpMode.linear, Evaluation.corpus)
    # xp(['EN'], Dataset.dimsum, XpMode.linear, Evaluation.corpus)

    # setOptimalRSGFeaturesForFtbSVM()
    # xp(['FR'], Dataset.ftb, XpMode.linear, Evaluation.corpus)
    # setOptimalRSGFeaturesForDimsumSVM()
    # xp(['EN'], Dataset.dimsum, XpMode.linear, Evaluation.corpus)

    setOptimalRSGFeaturesBySVM()
    setMinimalFeatures()
    configuration['others']['traitDeformedLemma'] = False
    configuration['others']['replaceNumbers'] = True
    xp(['TR'], Dataset.sharedtask2, XpMode.linear, Evaluation.corpus)
    #
    # setOptimalRSGForMLP()
    # xp(allSharedtask2Lang, Dataset.sharedtask2, None, Evaluation.corpus)
