import os

from enum import Enum

configuration = {
    'xp': {
        'linear': False,
        'compo': False,
        'pytorch': False,
        'kiperwasser': False,
        'rnn': False,
        'rnnNonCompo': False,
        'verbose': 1
    },
    'kiperwasser': {
        'wordDim': 25,
        'posDim': 5,
        'layerNum': 2,
        'activation': 'tanh',
        'optimizer': 'adam',
        'lr': 0.1,
        'dropout': .3,
        'epochs': 15,
        'batch': 1,
        'dense1': 25,
        'dense2': 0,
        'denseDropout': False,
        'lstmDropout': 0.3,
        'lstmLayerNum': 2,
        'focusedElemNum': 8,
        'lstmUnitNum': 8
    },
    'rnn': {
        'focusedElements': 7,
        'wordDim': 50,
        'posDim': 15,
        'gru': True,
        'wordRnnUnitNum': 16,
        'posRnnUnitNum': 5,
        'rnnDropout': .3,
        'useDense': True,
        'denseDropout': .3,
        'denseUnitNum': 25,
        'optimizer': 'adagrad',
        'lr': .05,
        'epochs': 20,
        'batchSize': 64,
        'earlyStop': True,
        'compactVocab': True,
        's0TokenNum': 4,
        's1TokenNum': 2,
        'bTokenNum': 1,
        'shuffle': False,
        'rnnSequence': False

    },
    'dataset': {
        'sharedtask2': False,
        'FTB': False,
        'dimsum': False
    },
    'sampling': {
        'overSampling': False,
        'importantSentences': False,
        'importantTransitions': False,
        'sampleWeight': False,
        'favorisationCoeff': 1,
        'mweRepeition': 35,
        'focused': False
    },
    'evaluation': {
        'cv': {'active': False,
               'currentIter': -1,
               'cvFolds': 5,
               },
        'corpus': False,
        'fixedSize': False,
        'dev': False,
        'trainVsDev': False,
        'trainVsTest': False,
        'tokenAvg': 270000,
        'testTokenAvg': 43000,
        'dataset': 'train',
        'debugTrainNum': 25,
        'test': 0.1,
        'load': False,
        'save': False,
        'shuffleTrain': False,
    },
    'preprocessing': {
        'data': {
            'shuffle': False,
            'universalPOS': True
        },
        'oracle': {
            'merge': 'right'
        }
    },
    'linear': {
        'svm': False,
        # Logistic regression otherwise
    },
    'model': {
        'inputItems': 4,
        'padding': {
            'active': False,
            's0Padding': 5,
            's1Padding': 5,
            'bPadding': 2,
        },
        'embedding': {
            'active': True,
            'concatenation': False,
            'posEmb': 25,
            'tokenEmb': 200,
            'usePos': True,
            'lemma': True,
            'initialisation': {
                'active': False,
                'modifiable': True,
                'oneHotPos': False,
                'pos': True,
                'token': True,
                'Word2VecWindow': 3,
                'type': 'frWac200'
                # 'dataFR.profiles.min.250'  # 'frWac200'
            },
            'frequentTokens': True,
            'compactVocab': False
        },
        'mlp': {
            'dense1': {
                'active': False,
                'unitNumber': 1024,
                'activation': 'relu',
                'dropout': 0.2
            },
            'dense2': {
                'active': False,
                'unitNumber': 512,
                'activation': 'relu',
                'dropout': 0.2
            }
        },
        'rnn': {
            'active': False,
            'gru': False,
            'stacked': False,
            'rnn1': {'unitNumber': 128,
                     'posUnitNumber': 32},
            'rnn2': {'unitNumber': 128}
        },
        'train': {
            'visualisation': {
                'batchStep': 50},
            'manipulateClassWeights': True,
            'optimizer': 'adagrad',
            'loss': 'categorical_crossentropy',
            'verbose': 0,
            'batchSize': 64,
            'epochs': 40,
            'earlyStop': True,
            'chickPoint': False,
            'validationSplit': .1,
            'monitor': 'val_loss',
            'minDelta': .2,
            'lr': 0.02
        },
        'predict': {
            'verbose': 0
        },
    },
    'features': {
        'active': False,
        'unigram': {
            'lemma': True,
            'token': True,
            'pos': True,
            'suffix': False,
            'b1': True
        },
        'bigram': {
            'active': True,
            's0b2': True
        },
        'trigram': True,
        'syntax': {
            'active': True,
            'abstract': True,
            'lexicalised': False,
            'bufferElements': 5
        },
        'dictionary': {
            'active': True,
            'mwt': True,
            's0TokenIsMWEToken': False,
            's0TokensAreMWE': False
        },
        'history': {
            '1': False,
            '2': False,
            '3': False
        },
        'stackLength': True,
        'distance': {
            's0s1': True,
            's0b0': True
        }
    },
    'path': {
        'results': 'Results',
        'projectPath': '',
        'corpusRelativePath': 'ressources/sharedtask'
    },
    'files': {
        'bestWeights': 'bestWeigths.hdf5',
        'train': {
            'conllu': 'train.conllu',
            'posAuto': 'train.conllu.autoPOS',
            'depAuto': 'train.conllu.autoPOS.autoDep'
        },
        'test': {
            'conllu': 'test.conllu',
            'posAuto': 'test.conllu.autoPOS',
            'depAuto': 'test.conllu.autoPOS.autoDep'
        },
        'embedding': {
            'frWac200': 'ressources/WordEmb/frWac/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin',
            'dataFR.profiles.min.250': 'ressources/WordEmb/dataFR.profiles.min',
            'frWiki50': 'ressources/WordEmb/vecs50-linear-frwiki'
        },
        'reports': {
            'summary': 'summary.json',
            'normaliser': 'normaliser.pkl',
            'config': 'setting.txt',
            'scores': 'scores.csv',
            'schema': 'schema.png',
            'history': 'history.pkl',
            'model': 'model.hdf5'
        }
    },
    'constants': {
        'unk': '*unknown*',
        'empty': '*empty*',
        'number': '*number*',
        'alpha': 0.5
    }

}


class Dataset(Enum):
    sharedtask2 = 0
    FTB = 1


class Evaluation(Enum):
    cv = 0
    corpus = 1
    fixedSize = 2
    dev = 3
    trainVsDev = 4
    trainVsTest = 5


class XpMode(Enum):
    linear = 0
    compo = 1
    pytorch = 2
    kiperwasser = 3
    rnn = 4
    rnnNonCompo = 5


def desactivateMainConf():
    configuration['features']['active'] = False

    configuration['model']['mlp']['dense1']['active'] = False
    configuration['model']['rnn']['active'] = False

    configuration['model']['embedding']['active'] = False
    configuration['model']['embedding']['initialisation']['active'] = False
    configuration['model']['embedding']['concatenation'] = False
    configuration['model']['embedding']['usePos'] = False


def resetFRStandardFeatures():
    featConf = configuration['features']

    featConf['unigram']['token'] = True
    featConf['unigram']['pos'] = True
    featConf['unigram']['lemma'] = True
    featConf['syntax']['active'] = True
    featConf['syntax']['abstract'] = True
    featConf['bigram']['s0b2'] = True
    featConf['bigram']['active'] = True
    featConf['trigram'] = False
    featConf['distance']['s0b0'] = True
    featConf['distance']['s0s1'] = True
    featConf['stackLength'] = False
    featConf['unigram']['b1'] = True
    featConf['dictionary']['active'] = True
    featConf['history']['1'] = False
    featConf['history']['2'] = False
    featConf['history']['3'] = False


def resetStandardFeatures(v=False):
    configuration['features'].update({
        'active': True,
        'unigram': {
            'lemma': v,
            'token': v,
            'pos': v,
            'suffix': v,
            'b1': v
        },
        'bigram': {
            'active': v,
            's0b2': v
        },
        'trigram': v,
        'syntax': {
            'active': v,
            'abstract': v,
            'lexicalised': v,
            'bufferElements': 5
        },
        'dictionary': {
            'active': v,
            'mwt': True,
            's0TokenIsMWEToken': v,
            's0TokensAreMWE': v
        },
        'history': {
            '1': v,
            '2': v,
            '3': v
        },
        'stackLength': v,
        'distance': {
            's0s1': v,
            's0b0': v
        }
    })


def setFeatureConf(active=True):
    resetFRStandardFeatures()
    configuration['features']['active'] = active


def setDense1Conf(active=True, unitNumber=128):
    configuration['model']['mlp']['dense1']['active'] = active
    configuration['model']['mlp']['dense1']['unitNumber'] = unitNumber


def setEmbConf(active=True, useToken=True, usePos=True, tokenEmb=200, posEmb=25, init=False):
    embConf = configuration['model']['embedding']
    embConf['active'] = active
    embConf['useToken'] = useToken
    embConf['tokenEmb'] = tokenEmb
    embConf['usePos'] = usePos
    embConf['posEmb'] = posEmb
    embConf['initialisation']['active'] = init


configuration['path']['projectPath'] = os.path.dirname(__file__)[:-len(os.path.basename(os.path.dirname(__file__)))]
