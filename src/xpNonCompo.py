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


def setOptimalRSGFeaturesForSVM():
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
    configuration['sampling'].update({
        'importantSentences': True,
        'overSampling': True,
        'sampleWeight': True,
        'favorisationCoeff': 6,
        'focused': True})

    configuration['mlp'].update({
        'lemma': True,
        'posEmb': 42,
        'tokenEmb': 480,
        'compactVocab': False,
        'dense1': True,
        'dense1UnitNumber': 58,
        'dense1Dropout': 0.429,
        'lr': 0.059,
        'optimizer': 'adagrad'})


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
    from rsg import createRSG, runRSG

    # oarsub -p "GPU<>'NO'" -q production -l nodes=1,walltime=90 "NNIdenSys/Scripts/nonCompo.sh" -n mlp1 -O Reports/mlp1 -E Reports/mlp1
    createRSG('mlp.p', None, xpNum=5000)
    runRSG(pilotLangs, Dataset.sharedtask2, None, Evaluation.fixedSize,
              fileName='mlp.p', xpNum=2, xpNumByThread=500)


    # createRSG('coop.mlpInLin', XpMode.linear)
    # runRSG(pilotLangs, Dataset.sharedtask2, None, Evaluation.fixedSize,
    #        xpNumByThread=100,
    #        fileName='coop.mlpInLin', xpNum=1, mlpInLinear=True)
    # Ev.linInMLP
    # configuration['mlp'].update({
    #     'posEmb': 78,
    #     'tokenEmb': 327,
    #     'dense1UnitNumber': 113,
    #     'dense1Dropout': 0.25,
    # })
    # xp(allSharedtask2Lang, Dataset.sharedtask2, None, Evaluation.corpus, linearInMlp=True)

    # NoSampling5
    # setOptimalRSGForMLP()
    # configuration['sampling'].update({
    #     'importantSentences': False,
    #     'overSampling': False,
    #     'sampleWeight': False,
    #     'favorisationCoeff': 6,
    #     'focused': False})
    # xp(allSharedtask2Lang, Dataset.sharedtask2, None, Evaluation.corpus, xpNum=10)

    #  emb.tune.1
    import rsg

    # configuration['mlp']['initialize'] = True
    # configuration['mlp']['tokenEmb'] = 300
    #
    # # rsg.createRSG('mlp.emb.p', None)
    # # rsg.runRSG(pilotLangs, Dataset.sharedtask2, None, Evaluation.fixedSize,
    # #            fileName='mlp.emb.p', xpNum=2, xpNumByThread=150)
    #
    # xp(['FR'], Dataset.sharedtask2, None, Evaluation.fixedSize)
