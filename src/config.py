import os

configuration = {
    'others': {
        'cvFolds': 5,
        'currentIter': -1,
        'shuffleTrain': False,
        'debugTrainNum': 25,
        'test': 0.2,
        'tokenAvg': 270000,
        'testTokenAvg': 43000,
        'mweRepeition': 35,
        'universalPOS': True,
        'svm': True,
        'svmScikit': True,
        'bufferElements': 5,
        'verbose': True,
        'deleteNumericalExpressions': False,
        'replaceNumbers': True,
        'removeFtbMWT': True,
        'printTest': False,
        'logReg':False,
        'svc':False,
        'traitDeformedLemma': True
    },
    'xp': {
        'linear': False,
        'compo': False,
        'kiperwasser': False,
        'kiperComp': False,
        'rnn': False,
        'rnnNonCompo': False,
    },
    'dataset': {
        'sharedtask2': False,
        'ftb': False,
        'dimsum': False
    },
    'evaluation': {
        'cv': False,
        'corpus': False,
        'fixedSize': False,
        'dev': False,
        'trainVsDev': False,
        'trainVsTest': False
    },
    'kiperwasser': {
        'wordDim': 25,
        'posDim': 5,
        'layerNum': 2,
        'optimizer': 'adagrad',
        'lr': 0.07,
        'dropout': .3,
        'epochs': 15,
        'batch': 1,
        'dense1': 25,
        'denseActivation': 'tanh',
        'denseDropout': 0,
        'rnnUnitNum': 8,
        'rnnDropout': 0.3,
        'rnnLayerNum': 2,
        'focusedElemNum': 8,
        'file': 'kiper.p',
        'earlyStop': False,
        'verbose': True,
        'eager': True,
        'gru': True,
        'earlyStopPatience': 7,
        'trainValidationSet': True

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
        's0TokenNum': 4,
        's1TokenNum': 2,
        'bTokenNum': 1,
        'shuffle': False,
        'rnnSequence': False,
        'earlyStopPatience': 7
    },
    'mlp': {
        'inputItems': 4,
        'posEmb': 42,
        'tokenEmb': 480,
        'lemma': True,
        'compactVocab': False,
        'optimizer': 'adagrad',
        'loss': 'categorical_crossentropy',
        'verbose': 0,
        'batchSize': 64,
        'epochs': 40,
        'earlyStop': True,
        'chickPoint': False,
        'validationSplit': .1,
        'minDelta': .2,
        'lr': 0.059,
        'dense1': True,
        'dense1UnitNumber': 60,
        'dense1Activation': 'relu',
        'dense1Dropout': 0.43,
        'dense2': False,
        'dense2UnitNumber': 0,
        'dense2Activation': 'relu',
        'dense2Dropout': 0,
        'predictVerbose': False,
        's0Padding': 5,
        's1Padding': 5,
        'bPadding': 2,
        'features': False
    },
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
    'sampling': {
        'overSampling': False,
        'importantSentences': False,
        'importantTransitions': False,
        'sampleWeight': False,
        'favorisationCoeff': 1,
        'focused': False
    },
    'features': {
        'lemma': True,
        'token': False,
        'pos': True,
        'suffix': False,
        'b1': False,
        's0b2': False,
        'bigram': True,
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
        'distanceS0b0': False,
        'numeric': False
    },
    'path': {
        'results': 'Results',
        'projectPath': '',
        'corpusFolder': 'ressources/sharedtask'
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

configuration['path']['projectPath'] = os.path.dirname(__file__)[:-len(os.path.basename(os.path.dirname(__file__)))]
