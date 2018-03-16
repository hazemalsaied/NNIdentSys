configuration = {
    "evaluation": {
        "train": False,
        "cv": {"active": False,
               "currentIter": -1,
               "cvFolds": 5
               },
        "corpus": False,
        "dataset": "train",
        "debug": True,
        "debugTrainNum": 400,
        "validation": 0.2,
        "load": False,
        "save": False,
        "cluster": True
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
        "embedding": {
            "active": True,
            "pos": True,
            "posEmb": 25,
            "tokenEmb": 200,
            "s0Padding": 5,
            "s1Padding": 5,
            "bPadding": 2,
            "initialisation": False,
            "lemma": False,
            "frequentTokens": True,
            "concatenation": True
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
            "chickPoint": False
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
            "1": True,
            "2": True,
            "3": True
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

import os

configuration["path"]["projectPath"] = os.path.dirname(__file__)[:-len(os.path.basename(os.path.dirname(__file__)))]


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
    print  res
    return res
# settings['projectPath'] = os.path.dirname(__file__)[:-len(os.path.basename(os.path.dirname(__file__)))]

# print settings["model"]["embedding"]["pos"]
# XP_DEBUG_DATA_SET = False
# XP_SMALL_DATA_SET = False
# XP_TRAIN_DATA_SET = False
# XP_CORPUS_DATA_SET = False
# XP_CROSS_VALIDATION = False
# 
# STACKED = False
# USE_GRU = False
# 
# XP_MERGE_ON_RIGHT = True
# 
# XP_LOAD_MODEL = False
# 
# XP_SAVE_MODEL = True
# 
# PROJECT_PATH = os.path.dirname(__file__)[:-len(os.path.basename(os.path.dirname(__file__)))]
# VEC_TRAIN_DATA_LABELS_NUM = None
# VEC_INPUT_DIMENSIONS = None
# 
# XP_SMALL_TRAIN_SENT_NUM = 4000
# XP_SMALL_TEST_SENT_NUM = 1000
# 
# CORPUS_USE_UNIVERSAL_POS_TAGS = True
# CORPUS_SHUFFLE = True
# 
# LSTM = False
# LSTM_INPUT_TIME_STEP = 4
# 
# MLP_USE_RANDOM_NORMAL_INIT = False
# MLP_ADD_SPARSE_FEATURES = True
# MLP_DROPOUT_1_VALUE = 0.2
# MLP_USE_TANH_2 = False
# MLP_USE_TANH_1 = False
# MLP_WORD_EMB_LIMIT = 200
# MLP_USE_VARIANCE_SCALING = False
# MLP_POS_EMB_LIMIT = 25
# MLP_LAYER_1_UNIT_NUM = 1024
# MLP = True
# MLP_USE_LEMMAS = False
# MLP_STACK_ELEMS_NUM = 2
# MLP_BUFFER_ELEMs_NUM = 2
# MLP_LAYER_2 = True
# MLP_USE_RELU_1 = True
# MLP_USE_RELU_2 = True
# MLP_DROPOUT_2_VALUE = 0.2
# MLP_USE_LOCAL_WORD_EMBEDDING = False
# MLP_LAYER_2_UNIT_NUM = 512
# 
# LSTM_1_UNIT_NUM = 512
# LSTM_2_UNIT_NUM = 512
# 
# MLP_DROPOUT_2 = False
# MLP_DROPOUT_1 = True
# MLP_USE_SIGMOID_2 = False
# 
# NN_VERBOSE = 2
# NN_PREDICT_VERBOSE = 0
# NN_BATCH_SIZE = 128
# NN_EPOCHS = 15
# NN_SHUFFLE = True
# 
# CV_CURRENT_ITERATION = -1
# CV_ITERATIONS = 5
# 
# XP_RESULT_PATH = '../Results/MWEFiles'
# 
# M1_USE_TOKEN = True
# 
# USE_POS_EMB = True
# 
# USE_CLUSTER = False
# USE_DENSE_3 = False
# USE_DENSE_2 = True
# 
# TRAINABLE_EMBEDDING = True
# INITIALIZE_EMBEDDING = True
# USE_ADAM = True
# 
# REMOVE_NON_FREQUENT_WORDS = True
# EARLY_STOP = True
# USE_MODEL_CHECK_POINT = False
# SAVE_WEIGHTS = False
# 
# USE_MODEL_LSTM_IN_TWO_DIRS = False
# USE_MODEL_MLP_Hyper_SIMPLE = False
# USE_MODEL_MLP_SIMPLE = False
# USE_MODEL_MLP_LSTM = False
# USE_MODEL_MLP_LSTM_2 = False
# USE_MODEL_MLP_AUX_SIMPLE = False
# USE_MODEL_CONF_TRANS = False
# PREDICT_VERBOSE = 0
# 
# USE_POS_EMB_MODULE = False
# USE_WORD_EMB_MODULE = False
# USE_SEPERATED_EMB_MODULE = False
# 
# PADDING_ON_S0 = 5
# PADDING_ON_S1 = 5
# PADDING_ON_B0 = 2
# 
# 
# def toString():
#     # pp = pprint.PrettyPrinter(indent=4)
#     lines = []
#     for key, value in globals().iteritems():
#         if key == key.upper():
#             lines.append(key.replace('_', ' ').lower() + ' = ' + str(value) + '\n')
#     lines.sort()
#     return ''.join(lines)
# 
# 
# def load(settingDic):
#     thisModule = sys.modules[__name__]
#     for key, value in settingDic.iteritems():
#         if key in globals():
#             setattr(thisModule, key, value)
