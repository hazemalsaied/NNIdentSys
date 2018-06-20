import cPickle as pickle
import errno
import os
import random
import sys

import numpy
from keras.models import load_model
from keras.utils import plot_model

from config import configuration

PATH_ROOT_REPORTS_DIR = os.path.join(configuration["path"]["projectPath"], 'Reports')

try:
    reportPath = os.path.join(configuration["path"]["projectPath"], PATH_ROOT_REPORTS_DIR)
    if not os.path.isdir(reportPath):
        os.makedirs(reportPath)
    schemaFolder = os.path.join(reportPath, 'schemas')
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
        sys.stdout.write('Result folder: {0}\n'.format(XP_CURRENT_DIR_PATH.split('/')[-1]))
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
        sys.stdout.write("# Schema file: {0}\n".format(rI))
        plot_model(model, to_file=shemaFile)
        return
    schemaPath = os.path.join(XP_CURRENT_DIR_PATH, repFiles["schema"])
    if evalConf["cv"]["active"]:
        schemaPath = os.path.join(XP_CURRENT_DIR_PATH, str(evalConf["cv"]["currentIter"]),
                                  repFiles["schema"])
    plot_model(model, to_file=schemaPath)
    sys.stdout.write('# Parameters = {0}\n'.format(model.count_params()))
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
    sys.stdout.write('# Load path = {0}\n'.format(modelPath))

    loaded_model.load_weights(os.path.join(loadFolderPath, 'weigth.hdf5'))
    # weightFie = os.path.join(LOADED_MODEL_PATH, MODEL_WEIGHT_FILE)
    # loaded_model.load_weights(weightFie)
    # loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model


def saveCVScores(scores):
    if not mustSave() or not evalConf["cv"]["active"]:
        return
    results = ''
    for i in range(len(scores)):
        if i == 0 or i % 4 == 0:
            tmpScores = '{0} : F-score: {1}, Rappel: {2}, Precsion: {3}\n'.format(
                scores[i], scores[i + 1] / evalConf["cv"]["currentIter"],
                           scores[i + 2] / evalConf["cv"]["currentIter"],
                           scores[i + 3] / evalConf["cv"]["currentIter"])
            if not tmpScores.startswith('0.0'):
                sys.stdout.write(tmpScores)
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
    sys.stdout.write("*" * 20 + '\n')
    sys.stdout.write("# XP = {0}\n".format(value))
    sys.stdout.write("*" * 20 + '\n')


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


featureNumLine = '# Feature number = '
linearParamLine = '# Feature number = '
paramLine = '# Parameters = '
scoreLine = '# F-Score(Ordinary) = 0'
linearTitleLine = '# Language = '
titleLine = '# XP = '


def mineLinearFile(newFile):
    path = '../Reports/Reports/{0}'.format(newFile)
    titles, params, scores = [], [], []
    with open(path, 'r') as log:
        for line in log.readlines():
            if line.startswith(linearParamLine):
                paramsValue = toNum(line[len(linearParamLine):len(linearParamLine) + 8].strip())
                params.append(round(int(paramsValue) / 1000000., 2))
            if line.startswith(scoreLine):
                fScore = toNum(line[len(scoreLine):len(scoreLine) + 5].strip())
                while len(fScore) < 4:
                    fScore = fScore + '0'
                scores.append(round(int(fScore) / 10000., 4) * 100)
            if line.startswith(linearTitleLine):
                titles.append(line[len(linearTitleLine):].strip())
    for i in range(len(scores)):
        if i < len(titles):
            print scores[i]  # titles[i]#, scores[i]
    return titles, scores, params


def mineFile(newFile):
    path = '../Reports/Reports/{0}'.format(newFile)
    titles, params, scores = [], [], []
    with open(path, 'r') as log:
        for line in log.readlines():
            if line.startswith(paramLine):
                paramsValue = toNum(line[len(paramLine):len(paramLine) + 8].strip())
                params.append(round(int(paramsValue) / 1000000., 2))
            if line.startswith(scoreLine):
                fScore = toNum(line[len(scoreLine):len(scoreLine) + 5].strip())
                while len(fScore) < 4:
                    fScore = fScore + '0'
                scores.append(round(int(fScore) / 10000., 4) * 100)
            if line.startswith(titleLine) and not line.startswith('WARNING:root:Title: Language : FR'):
                titles.append(line[len(titleLine):].strip())
    return titles, scores, params


def clean(titles, scores, params, xpNum):
    addedItems, newScores, newTitles, newParams = 0, [], [], []
    for i in range(1, len(scores)):
        if addedItems == xpNum:
            addedItems = 0
            continue
        newScores.append(scores[i])
        if i < len(params):
            newParams.append(params[i])
        if i < len(titles):
            newTitles.append(titles[i])
        addedItems += 1
    return newTitles, newScores, newParams


def divide(list, subListLength):
    stepNum, newList = len(list) / subListLength, []
    if stepNum:
        for i in range(stepNum):
            newList.append(list[i * subListLength: (i + 1) * subListLength])
        return newList
    return None


def getLinearScores(newFile):
    titles, scores, params = mineLinearFile(newFile)
    getDetailedScores(newFile, scores, titles, params)


def getScores(newFile, xpNum=10, shouldClean=False, showTitle=True):
    titles, scores, params = mineFile(newFile)
    if shouldClean:
        titles, scores, params = clean(titles, scores, params, xpNum)
    # getDetailedScores(newFile, scores, titles, params)
    getBrefScores(newFile, scores, titles, params, xpNum, showTitle=showTitle)


def getDetailedScores(newFile, scores, titles, params):
    text = '\\textbf{title}\t\t&\t\t\\textbf{F}\t\t\\textbf{P}\t\t\\\\\\hline\n'
    for i in range(len(scores)):
        titleText = titles[i] if i < len(titles) else ''
        paramsText = '\t\t&\t\t{0}\t'.format(params[i] if i < len(params) else '')
        paramsText = paramsText if paramsText != '\t\t&\t\t' else ''
        text += '{0}\t\t&\t\t{1}{2}\\\\\\hline\n'.format(titleText, scores[i], paramsText)
    with open('../Reports/{0}.detailed.csv'.format(newFile), 'w') as res:
        res.write(text)


def getBrefScores(newFile, scores, titles, params, xpNum, showTitle=True):
    scores = divide(scores, xpNum)
    params = divide(params, xpNum)
    titles = divide(titles, xpNum)
    # text = '\\textbf{title}\t&\t\\textbf{F$_{mean}$}\t&\t\\textbf{F$_{max}$}\t&' \
    #       '\t\t\\textbf{MAD}\t\t&\t\t\\textbf{P}\t\t\\\\\\hline\n'
    text = ''
    for i in range(len(scores)):
        # if i < 12:
        #     header = '{0}\t\t&\t\t{1}\t\t'.format(dom[i], '')
        # else:
        #     header= '{0}\t\t&\t\t{1}\t\t'.format('', dom[i])
        if showTitle:
            titleText = titles[i][0] if titles else ''
            if titleText:
                titleText += '\t\t\t\t&'
        else:
            titleText = ''
        population = scores[i]
        meanValue = round(numpy.mean(population), 1)
        maxValue = round(max(population), 1)
        mad = getMeanAbsoluteDeviation(population)
        paramsText = '\t\t&{0}\t\t'.format(round(numpy.mean(params[i]), 1) if params else '')
        paramsText = paramsText if paramsText != '\t\t&\t\t' else ''
        text += '{0}{1}\t\t&\t\t{2}\t\t&\t\t{3}{4}\t\t\\\\\n'.format(
            titleText, meanValue, maxValue, mad, paramsText)
    with open('../Reports/Reports/{0}.tex'.format(newFile), 'w') as res:
        res.write(text)


def toNum(text, addPoint=False):
    textBuf = ''
    for c in text:
        if c.isdigit():
            textBuf += c
    if addPoint and textBuf.strip() != '0':
        return float(textBuf) / (pow(10, len(textBuf)))
    return textBuf


def getMeanAbsoluteDeviation(domain):
    avg = numpy.mean(domain)
    distances = []
    for v in domain:
        dis = v - avg
        if dis < 0:
            dis *= -1
        distances.append(dis)
    return round(sum(distances) / len(distances), 1)


def attaachTwoFiles(f1, f2):
    res = ''
    with open(f1, 'r') as f1:
        with open(f2, 'r') as f2:
            idx = 0
            content = f2.readlines()
            for line1 in f1:
                res += line1[:-1] + content[idx]
                idx += 1
    print res


if __name__ == '__main__':
    # attaachTwoFiles('../Reports/Reports/1.txt' ,'../Reports/Reports/2.txt')
    # mineLinearFile('sharedtask2.min.txt')
    # getScores('tunning4', xpNum=5, showTitle=True, shouldClean=False)
    for f in os.listdir('../Reports/Reports'):
        print f
        if not f.endswith('.tex') and not f.lower().endswith('err'):
            getScores(f, shouldClean=False, showTitle=True, xpNum=5)
