import cPickle as pickle
import errno
import logging
import os
import random

from keras.models import load_model
from keras.utils import plot_model

from config import configuration

PATH_ROOT_REPORTS_DIR = os.path.join(configuration["path"]["projectPath"], 'Reports')

try:
    reportPath = os.path.join(configuration["path"]["projectPath"], PATH_ROOT_REPORTS_DIR)
    if not os.path.isdir(reportPath):
        os.makedirs(reportPath)
    schemaFolder =  os.path.join(reportPath, 'schemas')
    if not os.path.isdir(schemaFolder):
        os.makedirs(schemaFolder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

XP_CURRENT_DIR_PATH = ''
evalConf = configuration["evaluation"]
repFiles = configuration["files"]["reports"]


# def getXPDirectory(langName, xpNum):
#
#     if evalConf["load"] or evalConf["debug"]:
#         return
#     cvTxt = '-CV' if evalConf["cv"]["active"] else ''
#     prefix = langName + cvTxt + '-' + str(xpNum)
#     global XP_CURRENT_DIR_PATH
#     XP_CURRENT_DIR_PATH = os.path.join(reportPath, prefix)


# def createXPDirectory():
#     if evalConf["load"] or evalConf["debug"]:
#         return
#     try:
#         if not os.path.isdir(XP_CURRENT_DIR_PATH):
#             os.makedirs(XP_CURRENT_DIR_PATH)
#     except OSError as err:
#         if err.errno != errno.EEXIST:
#             raise


def createReportFolder(lang):
    if not mustSave():
        return
    xpNum = getXPNum()
    cvTxt = '-CV' if evalConf["cv"]["active"] else ''
    prefix = lang + cvTxt + '-' + str(xpNum)
    global XP_CURRENT_DIR_PATH
    XP_CURRENT_DIR_PATH = os.path.join(reportPath, prefix)
    if not evalConf["load"]:
        logging.warn('Result folder: {0}'.format(XP_CURRENT_DIR_PATH.split('/')[-1]))
        if not os.path.isdir(XP_CURRENT_DIR_PATH):
            os.makedirs(XP_CURRENT_DIR_PATH)


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


def saveModelSummary(model):
    if not mustSave():
        return
    json_string = model.to_json()
    summaryFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles["summary"])
    if evalConf["cv"]["active"]:
        summaryFile = os.path.join(XP_CURRENT_DIR_PATH, str(evalConf["cv"]["currentIter"]),
                                   repFiles["summary"])
    with open(summaryFile, 'a') as f:
        f.write(json_string)


def saveNormalizer(normalizer):
    if not mustSave():
        return
    vectFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles["normaliser"])
    if evalConf["cv"]["active"]:
        vectFile = os.path.join(XP_CURRENT_DIR_PATH, str(evalConf["cv"]["currentIter"]), repFiles["normaliser"])
    filehandler = open(vectFile, 'w')
    pickle.dump(normalizer, filehandler, pickle.HIGHEST_PROTOCOL)


# def saveSettings():
#     if evalConf["load"] or evalConf["debug"]:
#         return
#     if evalConf["cv"]["active"]:
#         settFile = os.path.join(XP_CURRENT_DIR_PATH, str(evalConf["cv"]["currentIter"]), SETTINGS_FILE)
#     else:
#         settFile = os.path.join(XP_CURRENT_DIR_PATH, SETTINGS_FILE)
#     settStr = settings.toString()
#     with open(settFile, 'w') as f:
#         f.write(settStr)


def saveScores(scores):
    if not mustSave():
        return
    results, line = '', ''
    for i in range(1, len(scores) + 1):
        line += str(scores[i - 1]) + ','
        if i % 4 == 0:
            results += line[:-1] + '\n'
            line = ''
    scoresFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles["scores"])
    if evalConf["cv"]["active"]:
        scoresFile = os.path.join(XP_CURRENT_DIR_PATH, str(evalConf["cv"]["currentIter"]), repFiles["scores"])
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
    normalizerPath = os.path.join(loadFolderPath, repFiles["normaliser"])
    return pickle.load(open(normalizerPath, "rb"))


def saveNetwork(model):
    if not mustSave():
        rI = random.randint(100, 500)
        shemaFile = os.path.join(configuration["path"]["projectPath"], 'Reports/schemas/shema{0}.png'.format(rI))
        logging.warn("Schema file: {0}".format(rI))
        plot_model(model, to_file=shemaFile)
        return
    schemaPath = os.path.join(XP_CURRENT_DIR_PATH, repFiles["schema"])
    if evalConf["cv"]["active"]:
        schemaPath = os.path.join(XP_CURRENT_DIR_PATH, str(evalConf["cv"]["currentIter"]),
                                  repFiles["schema"])
    plot_model(model, to_file=schemaPath)
    logging.warn('Parameter number: {0}'.format(model.count_params()))
    saveModelSummary(model)
    # saveModel(model)


def saveModel(model):
    if not mustSave():
        return
    if evalConf["cv"]["active"]:
        modelFile = os.path.join(XP_CURRENT_DIR_PATH, str(evalConf["cv"]["currentIter"]),
                                 repFiles["model"])
    else:
        modelFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles["model"])
    if not os.path.isfile(modelFile) and evalConf["save"]:
        model.save(modelFile)


def loadModel(loadFolderPath):
    modelPath = os.path.join(loadFolderPath, repFiles["model"])
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
    if not mustSave() or not evalConf["cv"]["active"]:
        return
    results = ''
    for i in range(len(scores)):
        if i == 0 or i % 4 == 0:
            tmpScores = '{0} : F-score: {1}, Rappel: {2}, Precsion: {3}'.format(
                scores[i], scores[i + 1] / evalConf["cv"]["currentIter"],
                           scores[i + 2] / evalConf["cv"]["currentIter"],
                           scores[i + 3] / evalConf["cv"]["currentIter"])
            if not tmpScores.startswith('0.0'):
                logging.warn(tmpScores)
                results += tmpScores + '\n'
    scoresFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles["scores"])
    with open(scoresFile, 'w') as f:
        f.write(results)


def settingsToDic():
    settFile = os.path.join(LOADED_MODEL_PATH, repFiles["config"])
    results = {}
    with open(settFile, 'r') as f:
        for line in f:
            parts = line.strip().split('=')
            if parts and len(parts) == 2:
                key = parts[0].strip().replace(' ', '_').upper()
                results[key] = parts[1]
    return results


def saveHistory(history):
    if not mustSave():
        return
    historyFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles["history"])
    if evalConf["cv"]["active"]:
        historyFile = os.path.join(XP_CURRENT_DIR_PATH, str(evalConf["cv"]["currentIter"]),
                                   repFiles["history"])
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
    if evalConf["cv"]["active"]:
        bestWeightPath = os.path.join(XP_CURRENT_DIR_PATH,
                                      str(evalConf["cv"]["currentIter"]),
                                      configuration["files"]["bestWeights"])
    return bestWeightPath


def mustSave():
    global evalConf
    evalConf = configuration["evaluation"]
    if evalConf["load"] or evalConf["debug"]:
        return False
    if evalConf["save"]:
        return True
    return False
