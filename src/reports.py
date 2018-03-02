import cPickle as pickle
import errno
import logging
import os

from keras.models import load_model
from keras.utils import plot_model

import settings

PATH_ROOT_REPORTS_DIR = os.path.join(settings.PROJECT_PATH, 'Reports')

try:
    reportPath = os.path.join(settings.PROJECT_PATH, PATH_ROOT_REPORTS_DIR)
    if not os.path.isdir(reportPath):
        os.makedirs(reportPath)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

XP_CURRENT_DIR_PATH = ''


def getXPDirectory(langName, xpNum):
    if settings.XP_LOAD_MODEL:
        return
    cvTxt = '-CV' if settings.XP_CROSS_VALIDATION else ''
    prefix = langName + cvTxt + '-' + str(xpNum)
    global XP_CURRENT_DIR_PATH
    XP_CURRENT_DIR_PATH = os.path.join(reportPath, prefix)


def createXPDirectory():
    if settings.XP_LOAD_MODEL:
        return
    try:
        if not os.path.isdir(XP_CURRENT_DIR_PATH):
            os.makedirs(XP_CURRENT_DIR_PATH)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise


MODEL_SUMMARY_FILE_NAME = 'summary.json'


def saveModelSummary(model):
    if settings.XP_LOAD_MODEL:
        return
    json_string = model.to_json()
    summaryFile = os.path.join(XP_CURRENT_DIR_PATH, MODEL_SUMMARY_FILE_NAME)
    if settings.XP_CROSS_VALIDATION:
        summaryFile = os.path.join(XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION), MODEL_SUMMARY_FILE_NAME)
    with open(summaryFile, 'a') as f:
        f.write(json_string)


NORMALIZER_OBLJ_FILE_NAME = 'normalizer.pkl'


def saveNormalizer(normalizer):
    if settings.XP_LOAD_MODEL:
        return
    vectFile = os.path.join(XP_CURRENT_DIR_PATH, NORMALIZER_OBLJ_FILE_NAME)
    if settings.XP_CROSS_VALIDATION:
        vectFile = os.path.join(XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION), NORMALIZER_OBLJ_FILE_NAME)
    filehandler = open(vectFile, 'w')
    pickle.dump(normalizer, filehandler, pickle.HIGHEST_PROTOCOL)


SETTINGS_FILE = 'setting.txt'


def saveSettings():
    if settings.XP_LOAD_MODEL:
        return
    if settings.XP_CROSS_VALIDATION:
        settFile = os.path.join(XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION), SETTINGS_FILE)
    else:
        settFile = os.path.join(XP_CURRENT_DIR_PATH, SETTINGS_FILE)
    settStr = settings.toString()
    with open(settFile, 'w') as f:
        f.write(settStr)


SCORES_FILE = 'scoers.csv'


def saveScores(scores):
    if settings.XP_LOAD_MODEL:
        return
    results, line = '', ''
    for i in range(1, len(scores) + 1):
        line += str(scores[i - 1]) + ','
        if i % 4 == 0:
            results += line[:-1] + '\n'
            line = ''
    scoresFile = os.path.join(XP_CURRENT_DIR_PATH, SCORES_FILE)
    if settings.XP_CROSS_VALIDATION:
        scoresFile = os.path.join(XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION), SCORES_FILE)
    with open(scoresFile, 'w') as f:
        f.write(results)


LOADED_MODEL_PATH = ''


def getLoadedModelPath(num, lang):
    prefix = lang + '-MLP-' if settings.MLP else '-LSTM-'
    loadedFolder = prefix + (
        ('0' + str(num)) if num < 10 else str(num))
    rPath = os.path.join(settings.PROJECT_PATH, PATH_ROOT_REPORTS_DIR)
    return os.path.join(rPath, loadedFolder)


def loadNormalizer(loadFolderPath):
    normalizerPath = os.path.join(loadFolderPath, NORMALIZER_OBLJ_FILE_NAME)
    return pickle.load(open(normalizerPath, "rb"))


MODEL_FILE = 'model.h5'
MODEL_WEIGHT_FILE = 'model.hdf5'
MODEL_SCHEMA = 'model.png'


def saveNetwork(model):
    if settings.XP_LOAD_MODEL:
        return
    schemaPath = os.path.join(XP_CURRENT_DIR_PATH, MODEL_SCHEMA)
    if settings.XP_CROSS_VALIDATION:
        schemaPath = os.path.join(XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION), MODEL_SCHEMA)
    plot_model(model, to_file=schemaPath)
    logging.warn('Parameter number: {0}'.format(model.count_params()))
    saveModelSummary(model)
    # saveModel(model)


def saveModel(model):
    if settings.XP_CROSS_VALIDATION:
        modelFile = os.path.join(XP_CURRENT_DIR_PATH, str(settings.CV_CURRENT_ITERATION), MODEL_WEIGHT_FILE)
    else:
        modelFile = os.path.join(XP_CURRENT_DIR_PATH, MODEL_WEIGHT_FILE)
    if not os.path.isfile(modelFile):
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
    if settings.XP_LOAD_MODEL or not settings.XP_CROSS_VALIDATION:
        return
    results = ''
    for i in range(len(scores)):
        if i == 0 or i % 4 == 0:
            tmpScores = '{0} : F-score: {1}, Rappel: {2}, Precsion: {3}'.format(
                scores[i], scores[i + 1] / settings.CV_ITERATIONS, scores[i + 2] / settings.CV_ITERATIONS,
                           scores[i + 3] / settings.CV_ITERATIONS)
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


def createHeader(title, value):
    #logging.warn("*" * 40)
    logging.warn("*" * 40)
    logging.warn("{0} {1}".format(title, value))
    logging.warn("*" * 40)
    #logging.warn("*" * 40)
