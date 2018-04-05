configuration = {
    "evaluation": {
        "train": False,
        "cv": {"active": False,
               "currentIter": -1,
               "cvFolds": 5,
               },
        "corpus": False,
        "dataset": "train",
        "debug": True,
        "debugTrainNum": 200,
        "test": 0.1,
        "load": False,
        "save": False,
        "cluster": True,
        "shuffleTrain": False
    },
    "preprocessing": {
        "data": {
            "shuffle": False,
            "universalPOS": True
        },
        "oracle": {
            "merge": "right"
        }
    },
    "model": {
        "padding": {
            "active": True,
            "s0Padding": 5,
            "s1Padding": 5,
            "bPadding": 2,
        },
        "embedding": {
            "active": True,
            "concatenation": True,
            "posEmb": 25,
            "tokenEmb": 200,
            "usePos": True,
            "lemma": False,
            "initialisation": {
                "active": False,
                "oneHotPos": False,
                "pos": True,
                "token": True,
                "Word2VecWindow": 3
            },
            "frequentTokens": True
        },
        "topology": {
            "mlp": {
                "active": True,
                "dense1": {
                    "active": False,
                    "unitNumber": 1024,
                    "activation": "relu",
                    "dropout": 0.2
                },
                "dense2": {
                    "active": False,
                    "unitNumber": 512,
                    "activation": "relu",
                    "dropout": 0.2
                },
                "dense3": {
                    "active": False,
                    "unitNumber": 512,
                    "activation": "relu",
                    "dropout": 0.2
                }
            },
            "rnn": {
                "active": False,
                "gru": False,
                "stacked": False,
                "rnn1": {"unitNumber": 512},
                "rnn2": {"unitNumber": 512}
            }
        },
        "train": {
            "optimizer": "ADAM",
            "loss": "categorical_crossentropy",
            "verbose": 2,
            "batchSize": 128,
            "epochs": 15,
            "earlyStop": True,
            "chickPoint": False,
            "validationSplit": .1,
            "monitor": 'val_loss',
            "minDelta": .1
        },
        "predict": {
            "verbose": 0
        },
    },
    "features": {
        "active": True,
        "unigram": {
            "lemma": True,
            "token": True,
            "pos": True,
            "suffix": False,
            "b1": True
        },
        "bigram": {
            "active": True,
            "s0b2": True
        },
        "trigram": True,
        "syntax": {
            "active": True,
            "abstract": True,
            "lexicalised": False,
            "bufferElements": 5
        },
        "dictionary": {
            "active": True,
            "mwt": False,
            "s0TokenIsMWEToken": True,
            "s0TokensAreMWE": False
        },
        "history": {
            "1": False,
            "2": False,
            "3": False
        }, "stackLength": True,
        "distance": {
            "s0s1": True,
            "s0b0": True
        }
    },
    "path": {
        "results": '/Results',
        "projectPath": '',
        'corpusRelativePath': "ressources/sharedtask/"
    },
    "files": {
        "bestWeights": "bestWeigths.hdf5",
        "train": {
            "conllu": "train.conllu",
            "posAuto": "train.conllu.autoPOS",
            "depAuto": "train.conllu.autoPOS.autoDep"
        }, "test": {
            "conllu": "test.conllu",
            "posAuto": "test.conllu.autoPOS",
            "depAuto": "test.conllu.autoPOS.autoDep"
        },
        "embedding": {
            "frWac200": "ressources/wordEmb/frWac/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
        }, "reports": {
            "summary": "'summary.json'",
            "normaliser": "normaliser.pkl",
            "config": "setting.txt",
            "scores": "scores.csv",
            "schema": "schema.png",
            "history": "history.pkl",
            "model": "model.hdf5"

        }

    },
    "constants": {
        "unk": "*unknown*",
        "empty": "*empty*",
        "number": "*number*",
        "alpha": 0.2
    }

}


def desactivateMainConf():
    configuration["features"]["active"] = False

    configuration["model"]["topology"]["mlp"]["active"] = False
    configuration["model"]["topology"]["mlp"]["dense1"] = False
    configuration["model"]["topology"]["rnn"]["active"] = False

    configuration["model"]["embedding"]["active"] = False
    configuration["model"]["embedding"]["initialisation"]["active"] = False
    configuration["model"]["embedding"]["concatenation"] = False
    configuration["model"]["embedding"]["usePos"] = False


def resetFRStandardFeatures():
    featConf = configuration["features"]

    featConf["unigram"]["token"] = True
    featConf["unigram"]["pos"] = True
    featConf["unigram"]["lemma"] = True
    featConf["syntax"]["active"] = True
    featConf["syntax"]["abstract"] = True
    featConf["bigram"]["s0b2"] = True
    featConf["bigram"]["active"] = True
    featConf["trigram"] = False
    featConf["distance"]["s0b0"] = True
    featConf["distance"]["s0s1"] = True
    featConf["stackLength"] = False
    featConf["unigram"]["b1"] = True
    featConf["dictionary"]["active"] = True
    featConf["history"]["1"] = False
    featConf["history"]["2"] = False
    featConf["history"]["3"] = False


def setFeatureConf(active=True):
    resetFRStandardFeatures()
    configuration["features"]["active"] = active


def setDense1Conf(active=True, unitNumber=128):
    configuration["model"]["topology"]["mlp"]["active"] = active
    configuration["model"]["topology"]["mlp"]["dense1"]["active"] = active
    configuration["model"]["topology"]["mlp"]["dense1"]["unitNumber"] = unitNumber


def setEmbConf(active=True, useToken=True, usePos=True, tokenEmb=200, posEmb=25, init=False):
    embConf = configuration["model"]["embedding"]
    embConf["active"] = active
    embConf["tokenEmb"] = tokenEmb
    embConf["posEmb"] = posEmb
    embConf["usePos"] = usePos
    embConf["useToken"] = useToken
    embConf["initialisation"]["active"] = init


import os

configuration["path"]["projectPath"] = os.path.dirname(__file__)[:-len(os.path.basename(os.path.dirname(__file__)))]


def printConf():
    # TODO
    evalConf = configuration["evaluation"]
    evalTxt = 'Dataset: '
    if evalConf["debug"]:
        evalTxt += ' Debug '
    if evalConf["train"]:
        evalTxt += ' Train '
    if evalConf["train"]:
        evalTxt += ' CV '
    evalTxt += '\n'

    paddingConf = configuration["model"]["padding"]
    paddingTxt = ''
    if paddingConf["active"]:
        paddingTxt += 'Padding is active, S0:{0} S1:{1} B:{2}'.format(
            paddingConf["s0Padding"], paddingConf["s1Padding"], paddingConf["bPadding"])
    embConf = configuration["model"]["embedding"]
    embTxt = ''
    if embConf["active"]:
        embTxt += 'Embedding is active, '

    pass


def getConfig():
    res = ''
    featConf = configuration["features"]
    if featConf["unigram"]["pos"] and featConf["unigram"]["lemma"]:
        res += 'A '
    if not featConf["unigram"]["pos"] and not featConf["unigram"]["lemma"]:
        res += 'A\''
    if featConf["syntax"]["active"]:
        res += 'B '
    if featConf["bigram"]["active"]:
        res += 'C '
    if featConf["trigram"]:
        res += 'D '
    if featConf["bigram"]["s0b2"]:
        res += 'E '
    if featConf["history"]["1"]:
        res += 'F '
    if featConf["history"]["2"]:
        res += 'G '
    if featConf["history"]["3"]:
        res += 'H '
    if featConf["distance"]["s0b0"]:
        res += 'I '
    if featConf["distance"]["s0s1"]:
        res += 'J '
    if featConf["unigram"]["b1"]:
        res += 'K '
    if featConf["dictionary"]:
        res += 'L '
    if featConf["stackLength"]:
        res += 'M '
    if featConf["dictionary"]["mwt"]:
        res += 'N '
    print res
    return res
