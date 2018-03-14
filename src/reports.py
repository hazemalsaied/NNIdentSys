import cPickle as pickle
import errno
import logging
import os

# from keras.models import load_model
from keras.utils import plot_model

from config import configuration

PATH_ROOT_REPORTS_DIR = os.path.join(configuration["path"]["projectPath"], 'Reports')

try:
    reportPath = os.path.join(configuration["path"]["projectPath"], PATH_ROOT_REPORTS_DIR)
    if not os.path.isdir(reportPath):
        os.makedirs(reportPath)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

XP_CURRENT_DIR_PATH = ''


def getXPDirectory(langName, xpNum):
    if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
        return
    cvTxt = '-CV' if configuration["evaluation"]["cv"]["active"] else ''
    prefix = langName + cvTxt + '-' + str(xpNum)
    global XP_CURRENT_DIR_PATH
    XP_CURRENT_DIR_PATH = os.path.join(reportPath, prefix)


def createXPDirectory():
    if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
        return
    try:
        if not os.path.isdir(XP_CURRENT_DIR_PATH):
            os.makedirs(XP_CURRENT_DIR_PATH)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise


def createReportFolder(lang):
    if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
        return
    xpNum = getXPNum()
    if configuration["model"]["train"]["save"]:
        getXPDirectory(lang, xpNum)
        if not configuration["evaluation"]["load"]:
            logging.warn('Result folder: {0}'.format(XP_CURRENT_DIR_PATH.split('/')[-1]))
            createXPDirectory()


def getXPNum():
    configPath = os.path.join(configuration["path"]["projectPath"], 'config.txt')
    with open(configPath, 'r+') as f:
        content = f.read()
        xpNum = int(content)
    with open(configPath, 'w') as f:
        newXpNum = xpNum + 1
        if newXpNum < 10:
            newXpNum = '000' + str(newXpNum)
        elif newXpNum < 100:
            newXpNum = '00' + str(newXpNum)
        elif newXpNum < 1000:
            newXpNum = '0' + str(newXpNum)
        else:
            newXpNum = str(newXpNum)
        f.write(newXpNum)
    return xpNum


MODEL_SUMMARY_FILE_NAME = 'summary.json'


def saveModelSummary(model):
    if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
        return
    json_string = model.to_json()
    summaryFile = os.path.join(XP_CURRENT_DIR_PATH, MODEL_SUMMARY_FILE_NAME)
    if configuration["evaluation"]["cv"]["active"]:
        summaryFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration["evaluation"]["cv"]["currentIter"]),
                                   MODEL_SUMMARY_FILE_NAME)
    with open(summaryFile, 'a') as f:
        f.write(json_string)


NORMALIZER_OBLJ_FILE_NAME = 'normalizer.pkl'


def saveNormalizer(normalizer):
    if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
        return
    vectFile = os.path.join(XP_CURRENT_DIR_PATH, NORMALIZER_OBLJ_FILE_NAME)
    if configuration["evaluation"]["cv"]["active"]:
        vectFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration["evaluation"]["cv"]["currentIter"]),
                                NORMALIZER_OBLJ_FILE_NAME)
    filehandler = open(vectFile, 'w')
    pickle.dump(normalizer, filehandler, pickle.HIGHEST_PROTOCOL)


SETTINGS_FILE = 'setting.txt'

# def saveSettings():
#     if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
#         return
#     if configuration["evaluation"]["cv"]["active"]:
#         settFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration["evaluation"]["cv"]["currentIter"]), SETTINGS_FILE)
#     else:
#         settFile = os.path.join(XP_CURRENT_DIR_PATH, SETTINGS_FILE)
#     settStr = settings.toString()
#     with open(settFile, 'w') as f:
#         f.write(settStr)


SCORES_FILE = 'scoers.csv'


def saveScores(scores):
    if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
        return
    results, line = '', ''
    for i in range(1, len(scores) + 1):
        line += str(scores[i - 1]) + ','
        if i % 4 == 0:
            results += line[:-1] + '\n'
            line = ''
    scoresFile = os.path.join(XP_CURRENT_DIR_PATH, SCORES_FILE)
    if configuration["evaluation"]["cv"]["active"]:
        scoresFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration["evaluation"]["cv"]["currentIter"]),
                                  SCORES_FILE)
    with open(scoresFile, 'w') as f:
        f.write(results)


LOADED_MODEL_PATH = ''


# def getLoadedModelPath(num, lang):
#     prefix = lang + '-MLP-' if settings.MLP else '-LSTM-'
#     loadedFolder = prefix + (
#         ('0' + str(num)) if num < 10 else str(num))
#     rPath = os.path.join(configuration["path"]["projectPath"], PATH_ROOT_REPORTS_DIR)
#     return os.path.join(rPath, loadedFolder)


def loadNormalizer(loadFolderPath):
    normalizerPath = os.path.join(loadFolderPath, NORMALIZER_OBLJ_FILE_NAME)
    return pickle.load(open(normalizerPath, "rb"))


MODEL_FILE = 'model.h5'
MODEL_WEIGHT_FILE = 'model.hdf5'
MODEL_SCHEMA = 'model.png'


def saveNetwork(model):
    if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
        return
    schemaPath = os.path.join(XP_CURRENT_DIR_PATH, MODEL_SCHEMA)
    if configuration["evaluation"]["cv"]["active"]:
        schemaPath = os.path.join(XP_CURRENT_DIR_PATH, str(configuration["evaluation"]["cv"]["currentIter"]),
                                  MODEL_SCHEMA)
    plot_model(model, to_file=schemaPath)
    logging.warn('Parameter number: {0}'.format(model.count_params()))
    saveModelSummary(model)
    # saveModel(model)


def saveModel(model):
    if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
        return
    if configuration["evaluation"]["cv"]["active"]:
        modelFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration["evaluation"]["cv"]["currentIter"]),
                                 MODEL_WEIGHT_FILE)
    else:
        modelFile = os.path.join(XP_CURRENT_DIR_PATH, MODEL_WEIGHT_FILE)
    if not os.path.isfile(modelFile) and configuration["model"]["train"]["save"]:
        model.save(modelFile)


def loadModel(loadFolderPath):
    modelPath = os.path.join(loadFolderPath, MODEL_WEIGHT_FILE)
    loaded_model = load_model(modelPath)
    logging.warn('Model is loaded from : {0}'.format(modelPath))

    loaded_model.load_weights(os.path.join(loadFolderPath, 'weigth.hdf5'))
    logging.warn('Model weights are loaded from : {0}'.format(os.path.join(loadFolderPath, 'weigth.hdf5')))

    # weightFie = os.path.join(LOADED_MODEL_PATH, MODEL_WEIGHT_FILE)
    # loaded_model.load_weights(weightFie)
    # loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    logging.warn('model loaded from disk!')
    logging.warn('No training needed!')
    return loaded_model


def saveCVScores(scores):
    if configuration["evaluation"]["load"] or not configuration["evaluation"]["cv"]["active"] or \
            configuration["evaluation"]["debug"]:
        return
    results = ''
    for i in range(len(scores)):
        if i == 0 or i % 4 == 0:
            tmpScores = '{0} : F-score: {1}, Rappel: {2}, Precsion: {3}'.format(
                scores[i], scores[i + 1] / configuration["evaluation"]["cv"]["currentIter"],
                           scores[i + 2] / configuration["evaluation"]["cv"]["currentIter"],
                           scores[i + 3] / configuration["evaluation"]["cv"]["currentIter"])
            if not tmpScores.startswith('0.0'):
                logging.warn(tmpScores)
                results += tmpScores + '\n'
    scoresFile = os.path.join(XP_CURRENT_DIR_PATH, SCORES_FILE)
    with open(scoresFile, 'w') as f:
        f.write(results)


def settingsToDic():
    settFile = os.path.join(LOADED_MODEL_PATH, SETTINGS_FILE)
    results = {}
    with open(settFile, 'r') as f:
        for line in f:
            parts = line.strip().split('=')
            if parts and len(parts) == 2:
                key = parts[0].strip().replace(' ', '_').upper()
                results[key] = parts[1]
    return results


def saveHistory(history):
    if configuration["evaluation"]["load"] or configuration["evaluation"]["debug"]:
        return
    historyFile = os.path.join(XP_CURRENT_DIR_PATH, 'history.pkl')
    if configuration["evaluation"]["cv"]["active"]:
        historyFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration["evaluation"]["cv"]["currentIter"]),
                                   'history.pkl')
    with open(historyFile, 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)


def createHeader(value):
    logging.warn("*" * 40)
    logging.warn("{0}".format(value))
    logging.warn("*" * 40)

def getBestWeightFilePath():
    if not configuration["model"]["train"]["chickPoint"]:
        return None
    bestWeightPath = os.path.join(XP_CURRENT_DIR_PATH, configuration["files"]["bestWeights"])
    if configuration["evaluation"]["cv"]["active"]:
        bestWeightPath = os.path.join(XP_CURRENT_DIR_PATH,
                                      str(configuration["evaluation"]["cv"]["currentIter"]),
                                      configuration["files"]["bestWeights"])
    return bestWeightPath