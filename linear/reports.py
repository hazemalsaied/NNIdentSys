import cPickle as pickle
import errno
import os
import random
import sys

import numpy
from keras.models import load_model
from keras.utils import plot_model

from config import configuration

tabs, seperator, doubleSep, finalLine = '\t', '\n' + '_' * 98 + '\n', '\n' + '=' * 98 + '\n', '\n' + '*|' * 49 + '\n'
PATH_ROOT_REPORTS_DIR = os.path.join(configuration["path"]["projectPath"], 'tmp')

try:
    reportPath = os.path.join(configuration["path"]["projectPath"], PATH_ROOT_REPORTS_DIR)
    if not os.path.isdir(reportPath):
        os.makedirs(reportPath)
        # schemaFolder = os.path.join(reportPath, 'schemas')
        # if not os.path.isdir(schemaFolder):
        #    os.makedirs(schemaFolder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

XP_CURRENT_DIR_PATH = ''
evalConf = configuration["evaluation"]
repFiles = configuration["files"]["reports"]


# def getXPDirectory(langName, xpNum):
#
#     if evalConf["load"]:
#         return
#     cvTxt = '-CV' if evalConf["cv"]["active"] else ''
#     prefix = langName + cvTxt + '-' + str(xpNum)
#     global XP_CURRENT_DIR_PATH
#     XP_CURRENT_DIR_PATH = os.path.join(reportPath, prefix)


# def createXPDirectory():
#     if evalConf["load"] :
#         return
#     try:
#         if not os.path.isdir(XP_CURRENT_DIR_PATH):
#             os.makedirs(XP_CURRENT_DIR_PATH)
#     except OSError as err:
#         if err.errno != errno.EEXIST:
#             raise


def printParsedSents(corpus, sentNum):
    printSentIdx, printSentNum = 0, sentNum
    for s in corpus.testingSents:
        if len(s.tokens) < 15 and len(s.vMWEs) == 1 and printSentIdx < printSentNum:
            sys.stdout.write(str(s) + '\n')
            printSentIdx += 1


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
#     if evalConf["load"]:
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
        shemaFile = os.path.join(configuration["path"]["projectPath"], 'tmp/schemas/shema{0}.png'.format(rI))
        sys.stdout.write("# Schema file: {0}\n".format(rI))
        plot_model(model, to_file=shemaFile)
        return
    schemaPath = os.path.join(XP_CURRENT_DIR_PATH, repFiles["schema"])
    if evalConf["cv"]["active"]:
        schemaPath = os.path.join(XP_CURRENT_DIR_PATH, str(evalConf["cv"]["currentIter"]),
                                  repFiles["schema"])
    plot_model(model, to_file=schemaPath)
    # sys.stdout.write('# Parameters = {0}\n'.format(model.count_params()))
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
    if value:
        sys.stdout.write(doubleSep + doubleSep + tabs + "{0}\n".format(value) + doubleSep)


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
    if evalConf["load"]:
        return False
    if evalConf["save"]:
        return True
    return False


def printMode():
    evalConf = configuration['evaluation']
    res = 'Debug'
    if evalConf['dev']:
        res = 'Train vs Dev'
    elif evalConf['corpus']:
        res = 'Train + Dev vs Test'
    elif evalConf['fixedSize']:
        res = 'Fixed Size'
    elif evalConf['trainVsTest']:
        res = 'Train vs Test'
    if res:
        sys.stdout.write(doubleSep + tabs + '{0} Mode'.format(res) + doubleSep)


featureNumLine = 'Trainable params: '
linearParamLine = '# Feature number = '
paramLine = 'Trainable params: '
scoreLine = '	Identification : 0'  # tabs + 'Identification : 0'
linearTitleLine = '# Language = '
titleLine = '# XP = '


def mineLinearFile(newFile):
    path = '../tmp/tmp/{0}'.format(newFile)
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
    path = '../tmp/tmp/{0}'.format(newFile)
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
    with open('../tmp/{0}.detailed.csv'.format(newFile), 'w') as res:
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
        print meanValue
    with open('../tmp/tmp/{0}.tex'.format(newFile), 'w') as res:
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


def getStats(newFile):
    path = '../tmp/tmp/{0}'.format(newFile)
    langs, mweLEngth, oldMWEs, newMWEs, params, dataSize, scores, correctlyIdentifiedList, nonIdentifiedList \
        = [], [], [], [], [], [], [], [], []
    langLine = tabs + 'Language : '
    mweLengthLine = tabs + 'MWE length mean : '
    oldMWEsLine = tabs + 'Seen MWEs : '
    newMWEsLine = tabs + 'New MWEs : '
    paramLine = 'Total params: '
    dataSizeLine = tabs + 'data size after sampling = '
    scoreLine = tabs + 'Identification : '
    correctlyIdentifiedLine = tabs + 'Correctly identified MWEs'
    nonIdentifiedLine = 'Non Identified MWEs'
    correctlyIdentified, nonIdentified = False, False
    nonIdentifiedDic, nonIdentifiedDic = dict(), dict()
    with open(path, 'r') as log:
        for line in log:
            line = line[:-1]
            if line.startswith(langLine):
                langs.append(line[len(langLine):].strip())
            elif line.startswith(mweLengthLine):
                mweLEngth.append(line[len(mweLengthLine):].strip())
            elif line.startswith(oldMWEsLine):
                oldMWEs.append(line[len(oldMWEsLine) + 5:-3].strip())
            elif line.startswith(newMWEsLine):
                newMWEs.append(line[len(newMWEsLine) + 5:-3].strip())
            elif line.startswith(paramLine):
                params.append(line[len(paramLine):].strip())
            elif line.startswith(dataSizeLine):
                dataSize.append(line[len(dataSizeLine):].strip())
            elif line.startswith(scoreLine):
                scores.append(round(float(line[len(scoreLine):].strip()) * 100, 2))

            if line.startswith(correctlyIdentifiedLine):
                correctlyIdentified = True
                correctlyIdentifiedDic = {}
            if correctlyIdentified and line.strip() and not line.startswith('-') and not line.startswith('='):
                line = line.strip()
                parts = line.split(' : ')
                if len(parts) == 2:
                    correctlyIdentifiedDic[parts[0]] = parts[1]

            if nonIdentifiedLine in line:  # line.startswith(nonIdentifiedLine):
                correctlyIdentified = False
                correctlyIdentifiedList.append(correctlyIdentifiedDic)
                correctlyIdentifiedDic = dict()
                nonIdentified = True
                nonIdentifiedDic = dict()

            if nonIdentified and line.strip() and not line.startswith('-') and not line.startswith('='):
                line = line.strip()
                parts = line.split(' : ')
                if len(parts) == 2:
                    nonIdentifiedDic[parts[0]] = parts[1]

            if line.startswith('*|'):
                nonIdentified = False
                nonIdentifiedList.append(nonIdentifiedDic)
    for i in range(len(correctlyIdentifiedList)):
        for k in correctlyIdentifiedList[i].keys():
            if int(correctlyIdentifiedList[i][k]) < 5:
                del correctlyIdentifiedList[i][k]
        for k in nonIdentifiedList[i].keys():
            if int(nonIdentifiedList[i][k]) < 3:
                del nonIdentifiedList[i][k]
    for i in range(len(correctlyIdentifiedList)):
        nonIdentifiedList[i] = str(nonIdentifiedList[i]).replace(',', '-')
        correctlyIdentifiedList[i] = str(correctlyIdentifiedList[i]).replace(',', '-')
        params[i] = params[i].replace(',', ' ')
        scores[i] = str(scores[i]).replace(',', '.')
    res = ''

    for i in range(len(scores)):
        res += '{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(langs[i], scores[i], mweLEngth[i], oldMWEs[i], newMWEs[i],
                                                              params[i], dataSize[i], correctlyIdentifiedList[i],
                                                              nonIdentifiedList[i])
    with open(path + '.csv', 'w') as f:
        f.write(res)
    print res


langLine1 = '	Language : '
langLine = '	GPU Enabled	Language : '

def getAvgScores(scores, langNum=3, trialNum=3):
    result, xpScores, langSum = [], [], 0
    for i, v in enumerate(scores):
        if i != 0 and i % trialNum == 0:
            xpScores.append(round(float(langSum) / trialNum, 1))
            langSum = 0
        if i != 0 and i % (trialNum * langNum) == 0:
            result.append(xpScores)
            xpScores = []
        langSum += float(v)
    xpScores.append(round(float(langSum) / trialNum, 1))
    result.append(xpScores)
    return result


def getNewScores(files):
    for f in files:
        f = str(f)
        titles, scores, params, langs, titles2 = mineNewFile(f)
        # results = getAvgScores(scores, 6)
        # orderedScores = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        for i, v in enumerate(titles):
            print v
            # print v, titles[i][:-1], '\n'  # ',', , ',', v
        # for i in range(len(titles)):
        #     print \
        #         scores[i * 3] if i * 3 < len(scores) else '', ',', \
        #         scores[i * 3 + 1] if i * 3 + 1 < len(scores) else '', ',', \
        #         scores[i * 3 + 2] if i * 3 + 2 < len(scores) else '', ',', \
        #         titles[i][:-1]
            # for i, t in enumerate(scores):

            #   print scores[i], titles[i]
            # t = t.replace(',', '.').replace('_', ',')
            # print f, ',', \
            #    t, str(results[i]).replace('[', ',').replace(']', ','), round(sum(results[i]) / 6, 1)
            # ii = i * 9 * 2
            # str(scores[ii:ii + 3]).replace('[', ',').replace(']', ','), \
            # results[i][0], \
            # str(scores[ii + 3:ii + 6]).replace('[', ',').replace(']', ','), \
            # results[i][1], \
            # str(scores[ii + 6:ii + 9]).replace('[', ',').replace(']', ','), \
            # results[i][2], ',', \
            # round((results[i][0] + results[i][1] + results[i][2]) / 3, 2)


def mineNewFile(newFile):
    path = '../tmp/tmp/{0}'.format(newFile)
    titles, params, scores, langs, titles2 = [], [], [], [], []
    with open(path, 'r') as log:
        for line in log.readlines():
            if line.startswith('validation loss after '):
                titles.append(line)
            if line.startswith(langLine) or line.startswith(langLine1):
                if line.startswith(langLine):
                    langs.append(line[len(langLine):len(langLine) + 2])
                else:
                    langs.append(line[len(langLine1):len(langLine1) + 2])
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
    return titles, scores, params, langs, titles2


if __name__ == '__main__':
    # attaachTwoFiles('../tmp/tmp/1.txt' ,'../tmp/tmp/2.txt')
    # mineLinearFile('sharedtask2.min.txt')
    getNewScores([
        'k.epochs'
        # 'kc.bg.rsg.5', 'kc.bg.rsg.6', 'kc.bg.rsg.7', 'kc.bg.rsg.8' , 'kc.bg.rsg.9', 'kc.bg.rsg.10'
        # 'bg.rsg.1', 'bg.rsg.2', 'bg.rsg.3', 'bg.rsg.4', 'bg.rsg.5','bg.rsg.6', 'bg.rsg.7', 'bg.rsg.8', 'bg.rsg.9',
        # 'bg.rsg.10', 'bg.rsg.11','bg.rsg.12', 'bg.rsg.13', 'bg.rsg.14', 'bg.rsg.15', 'bg.rsg.16', 'bg.rsg.17',
        # 'bg.rsg.18'
        # 'RNN.EVAL'
        # 'rnn.pilot.1', 'rnn.pilot.2', 'rnn.pilot.3', 'rnn.pilot.4', 'rnn.pilot.5', 'rnn.pilot.6',
        # 'rnn.pilot.7', 'rnn.pilot.8', 'rnn.pilot.9', 'rnn.pilot.10', 'rnn.pilot.11', 'rnn.pilot.12'
        # 'rnn.extra.sam'
        # 'linear1', 'linear1', 'linear13', 'linear14', 'linear15',
        # 'linear16', 'linear17', 'linear18', 'linear19', 'linear20',
        # 'linear11', 'linear12', 'linear13', 'linear14', 'linear15',
        # 'linear16', 'linear17', 'linear18', 'linear19', 'linear20'
    ])
    # getStats('earlyStopping.st2.corpus')
    # getScores('sharedtask2.new', xpNum=1, showTitle=True, shouldClean=False)
    # for f in os.listdir('../tmp/tmp'):
    #     print f
    #     if not f.endswith('.tex') and not f.lower().endswith('err'):
    #         getScores(f, shouldClean=False, showTitle=True, xpNum=5)
