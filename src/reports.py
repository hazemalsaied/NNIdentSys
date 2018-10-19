import cPickle as pickle
import errno
import os
import sys

import numpy

from config import configuration

tabs, seperator, doubleSep, finalLine = '\t', '\n' + '_' * 98 + '\n', '\n' + '=' * 98 + '\n', '\n' + '*|' * 49 + '\n'
PATH_ROOT_REPORTS_DIR = os.path.join(configuration['path']['projectPath'], 'Reports')

try:
    reportPath = os.path.join(configuration['path']['projectPath'], PATH_ROOT_REPORTS_DIR)
    if not os.path.isdir(reportPath):
        os.makedirs(reportPath)
        # schemaFolder = os.path.join(reportPath, 'schemas')
        # if not os.path.isdir(schemaFolder):
        #    os.makedirs(schemaFolder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

XP_CURRENT_DIR_PATH = ''
configuration['evaluation'] = configuration['evaluation']
repFiles = configuration['files']['reports']


# def getXPDirectory(langName, xpNum):
#
#     if configuration['evaluation']['load']:
#         return
#     cvTxt = '-CV' if configuration['evaluation']['cv'] else ''
#     prefix = langName + cvTxt + '-' + str(xpNum)
#     global XP_CURRENT_DIR_PATH
#     XP_CURRENT_DIR_PATH = os.path.join(reportPath, prefix)


# def createXPDirectory():
#     if configuration['evaluation']['load'] :
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
    xpNum = getXPNum()
    cvTxt = '-CV' if configuration['evaluation']['cv'] else ''
    prefix = lang + cvTxt + '-' + str(xpNum)
    global XP_CURRENT_DIR_PATH
    XP_CURRENT_DIR_PATH = os.path.join(reportPath, prefix)
    if not configuration['evaluation']['load']:
        sys.stdout.write('Result folder: {0}\n'.format(XP_CURRENT_DIR_PATH.split('/')[-1]))
        if not os.path.isdir(XP_CURRENT_DIR_PATH):
            os.makedirs(XP_CURRENT_DIR_PATH)


def getXPNum():
    configPath = os.path.join(configuration['path']['projectPath'], 'config.txt')
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


# def saveModelSummary(model):
#     if not mustSave():
#         return
#     json_string = model.to_json()
#     summaryFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles['summary'])
#     if configuration['evaluation']['cv']:
#         summaryFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration['others']['currentIter']),
#                                    repFiles['summary'])
#     with open(summaryFile, 'a') as f:
#         f.write(json_string)


# def saveNormalizer(normalizer):
#     if not mustSave():
#         return
#     vectFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles['normaliser'])
#     if configuration['evaluation']['cv']:
#         vectFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration['others']['currentIter']),
#                                 repFiles['normaliser'])
#     filehandler = open(vectFile, 'w')
#     pickle.dump(normalizer, filehandler, pickle.HIGHEST_PROTOCOL)


# def saveSettings():
#     if configuration['evaluation']['load']:
#         return
#     if configuration['evaluation']['cv']:
#         settFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration['others']['currentIter']), SETTINGS_FILE)
#     else:
#         settFile = os.path.join(XP_CURRENT_DIR_PATH, SETTINGS_FILE)
#     settStr = settings.toString()
#     with open(settFile, 'w') as f:
#         f.write(settStr)


# def saveScores(scores):
#     if not mustSave():
#         return
#     results, line = '', ''
#     for i in range(1, len(scores) + 1):
#         line += str(scores[i - 1]) + ','
#         if i % 4 == 0:
#             results += line[:-1] + '\n'
#             line = ''
#     scoresFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles['scores'])
#     if configuration['evaluation']['cv']:
#         scoresFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration['others']['currentIter']), repFiles['scores'])
#     with open(scoresFile, 'w') as f:
#         f.write(results)


LOADED_MODEL_PATH = ''


# def getLoadedModelPath(num, lang):
#     prefix = lang + '-MLP-' if settings.MLP else '-LSTM-'
#     loadedFolder = prefix + (
#         ('0' + str(num)) if num < 10 else str(num))
#     rPath = os.path.join(configuration['path']['projectPath'], PATH_ROOT_REPORTS_DIR)
#     return os.path.join(rPath, loadedFolder)


def loadNormalizer(loadFolderPath):
    normalizerPath = os.path.join(loadFolderPath, repFiles['normaliser'])
    return pickle.load(open(normalizerPath, 'rb'))


# def saveNetwork(model):
# if not mustSave():
# rI = random.randint(100, 500)
# shemaFile = os.path.join(configuration['path']['projectPath'], 'tmp/schemas/shema{0}.png'.format(rI))
# sys.stdout.write('# Schema file: {0}\n'.format(rI))
# plot_model(model, to_file=shemaFile)
# return
# schemaPath = os.path.join(XP_CURRENT_DIR_PATH, repFiles['schema'])
# if configuration['evaluation']['cv']:
#     schemaPath = os.path.join(XP_CURRENT_DIR_PATH, str(configuration['others']['currentIter']),
#                               repFiles['schema'])
# plot_model(model, to_file=schemaPath)
# sys.stdout.write('# Parameters = {0}\n'.format(model.count_params()))
# saveModelSummary(model)
# saveModel(model)


# def saveModel(model):
#     if not mustSave():
#         return
#     if configuration['evaluation']['cv']:
#         modelFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration['others']['currentIter']),
#                                  repFiles['model'])
#     else:
#         modelFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles['model'])
#     if not os.path.isfile(modelFile) and configuration['evaluation']['save']:
#         model.save(modelFile)


# def loadModel(loadFolderPath):
#     modelPath = os.path.join(loadFolderPath, repFiles['model'])
#     loaded_model = load_model(modelPath)
#     sys.stdout.write('# Load path = {0}\n'.format(modelPath))
#
#     loaded_model.load_weights(os.path.join(loadFolderPath, 'weigth.hdf5'))
#     # weightFie = os.path.join(LOADED_MODEL_PATH, MODEL_WEIGHT_FILE)
#     # loaded_model.load_weights(weightFie)
#     # loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     return loaded_model


# def saveCVScores(scores):
#     if not mustSave() or not configuration['evaluation']['cv']:
#         return
#     results = ''
#     for i in range(len(scores)):
#         if i == 0 or i % 4 == 0:
#             tmpScores = '{0} : F-score: {1}, Rappel: {2}, Precsion: {3}\n'.format(
#                 scores[i], scores[i + 1] / configuration['others']['currentIter'],
#                            scores[i + 2] / configuration['others']['currentIter'],
#                            scores[i + 3] / configuration['others']['currentIter'])
#             if not tmpScores.startswith('0.0'):
#                 sys.stdout.write(tmpScores)
#                 results += tmpScores + '\n'
#     scoresFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles['scores'])
#     with open(scoresFile, 'w') as f:
#         f.write(results)


def settingsToDic():
    settFile = os.path.join(LOADED_MODEL_PATH, repFiles['config'])
    results = {}
    with open(settFile, 'r') as f:
        for line in f:
            parts = line.strip().split('=')
            if parts and len(parts) == 2:
                key = parts[0].strip().replace(' ', '_').upper()
                results[key] = parts[1]
    return results


# def saveHistory(history):
#     if not mustSave():
#         return
#     historyFile = os.path.join(XP_CURRENT_DIR_PATH, repFiles['history'])
#     if configuration['evaluation']['cv']:
#         historyFile = os.path.join(XP_CURRENT_DIR_PATH, str(configuration['others']['currentIter']),
#                                    repFiles['history'])
#     with open(historyFile, 'wb') as f:
#         pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)


def createHeader(value):
    if value:
        sys.stdout.write(doubleSep + doubleSep + tabs + '{0}\n'.format(value) + doubleSep)


def getBestWeightFilePath():
    if not configuration['mlp']['chickPoint']:
        return None
    bestWeightPath = os.path.join(XP_CURRENT_DIR_PATH, configuration['files']['bestWeights'])
    if configuration['evaluation']['cv']:
        bestWeightPath = os.path.join(XP_CURRENT_DIR_PATH,
                                      str(configuration['others']['currentIter']),
                                      configuration['files']['bestWeights'])
    return bestWeightPath


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


def getNewScores(files, pilot=False, withTitles=True, withTitle2=False):
    for f in files:
        f = str(f)
        titles, scores, params, langs, titles2 = mineNewFile(f)
        if pilot:
            for i in range(len(titles)):
                if withTitle2:
                    print 'BG, PT, TR, ' + titles2[i][:-1]
                print '{0},{1},{2},{3}'.format(
                    scores[i * 3] if i * 3 < len(scores) else '',
                    scores[i * 3 + 1] if i * 3 + 1 < len(scores) else '',
                    scores[i * 3 + 2] if i * 3 + 2 < len(scores) else '',
                    titles[i][:-1] if withTitles else '')
        else:
            for i, v in enumerate(scores):
                print v, titles[i][:-1] if withTitles else ''


def mineNewFile(newFile):
    path = '../Reports/Reports/{0}'.format(newFile)
    titles, params, scores, langs, titles2 = [], [], [], [], []
    previousLine = ''
    with open(path, 'r') as log:
        for line in log.readlines():
            if line.startswith('Total loss for epoch '):
                print line[len('Total loss for epoch ') + 2:]
                continue
            if line.startswith('# Configs:'):
                titles.append(line)
            if line.startswith('# CTitles:'):
                titles2.append(line)
            if line.startswith(langLine) or line.startswith(langLine1):
                if line.startswith(langLine):
                    langs.append(line[len(langLine):len(langLine) + 2])
                else:
                    langs.append(line[len(langLine1):len(langLine1) + 2])
            if line.startswith(paramLine):
                paramsValue = toNum(line[len(paramLine):len(paramLine) + 8].strip())
                params.append(round(int(paramsValue) / 1000000., 2))
            if line.startswith(scoreLine) and previousLine.startswith('='):
                fScore = toNum(line[len(scoreLine):len(scoreLine) + 5].strip())
                while len(fScore) < 4:
                    fScore = fScore + '0'
                scores.append(round(int(fScore) / 10000., 4) * 100)
            if line.startswith(titleLine) and not line.startswith('WARNING:root:Title: Language : FR'):
                titles.append(line[len(titleLine):].strip())
            previousLine = line
    return titles, scores, params, langs, titles2


def orderResults(titles, values):
    results = dict()
    for i in range(len(titles)):
        xpTitles = titles[i].split(',')
        xpValues = values[i].split(',')
        if len(xpTitles) != len(xpValues):
            pass
        results[i] = dict()
        for j in range(len(xpTitles)):
            results[i][xpTitles[j]] = xpValues[j]
    for i in range(len(titles)):
        print sorted(results[i].items())

def readAndOrderResults():
    titles, values, idx = [], [], 0

    with open( 'tmp - tmp.csv.csv', 'r') as f:
        for line in f:
            if idx % 2 == 0:
                titles.append(line[:-1])
            else:
                values.append(line[:-1])
            idx += 1
    orderResults(titles, values)

if __name__ == '__main__':
    getNewScores([
        'mlpInLinear'
        #'svm.1', 'svm.2', 'svm.3', 'svm.4', 'svm.5', 'svm.6', 'svm.7'
        # 'k.rsg1', 'k.rsg2', 'k.rsg3','k.rsg4', 'k.rsg5', 'k.rsg6','k.rsg7', 'k.rsg8', 'k.rsg9','k.rsg10'
        # 'kc.bg.rsg.5', 'kc.bg.rsg.6', 'kc.bg.rsg.7', 'kc.bg.rsg.8' , 'kc.bg.rsg.9', 'kc.bg.rsg.10'
        # 'rnn.pilot.7', 'rnn.pilot.8', 'rnn.pilot.9', 'rnn.pilot.10', 'rnn.pilot.11', 'rnn.pilot.12'
        # 'linear16', 'linear17', 'linear18', 'linear19', 'linear20'
    ], pilot=False, withTitles=False, withTitle2=False)

