from statsmodels.stats.contingency_tables import mcnemar

from evaluation import evaluate
from xpTools import *

testSetKey, misIdentifiedKey, nonIdentifiedKey, corIdentifiedKey = 'Test set', 'Misidentified', 'Non identified', \
                                                                   'Correctly identified'
mweKey, mwtKey, partSeenKey, seenKey, barelySeenKey = 'mwe', '0.3 mwt', '2.1 Seen : partially', '2.3 Seen : all', \
                                                      '2.2 Seen : barely'
interleavingKey, embeddedKey, newKey, freqSeenKey = '0.1 Interleaving', '0.2 Embedded', \
                                                    '1.1 New', '2.3 Seen : frequently'


def analyzeCorpus(xpMode, dataset, division):
    for l in allSharedtask2Lang:
        testFileName = '12.11.fixedsize.{0}.cupt'.format(l)
        setXPMode(xpMode)
        corpus = readIdentifiedCorpus(l, dataset, division, testFileName)
        evaluate(corpus.testingSents)
        catAnalysis = getCatAnalysis(corpus)
        analysis = getErrorAnalysis(corpus)
        report(analysis, catAnalysis, l)


def getErrorAnalysisTable(xpMode, dataset, division):
    table = []
    for lang in allSharedtask2Lang:
        testFileName = '12.11.fixedsize.{0}.cupt'.format(lang)
        setXPMode(xpMode)
        corpus = readIdentifiedCorpus(lang, dataset, division, testFileName)
        evaluate(corpus.testingSents)
        analysis = getErrorAnalysis(corpus)
        line = [getOccurrences(analysis[corIdentifiedKey][mweKey]),
                getOccurrences(analysis[nonIdentifiedKey][mweKey])]
        for k in [corIdentifiedKey, misIdentifiedKey, nonIdentifiedKey]:
            line += [getOccurrences(analysis[k][c]) for c in
                     [barelySeenKey, freqSeenKey, partSeenKey, newKey]]
        table.append(line)
    return table


def getCatAnalysisTable(xpMode, dataset, division):
    table = []
    for lang in allSharedtask2Lang:
        if lang == 'IT':
            pass
        testFileName = '12.11.fixedsize.{0}.cupt'.format(lang)
        setXPMode(xpMode)
        corpus = readIdentifiedCorpus(lang, dataset, division, testFileName)
        evaluate(corpus.testingSents)
        catAnalysis = getCatAnalysis(corpus)
        line = []
        allOccurrences = 0.
        for d in catAnalysis.keys():
            allOccurrences += catAnalysis[d]['true'] + catAnalysis[d]['false']
        for d in sorted(catAnalysis.keys()):
            if catAnalysis[d]['true'] + catAnalysis[d]['false'] != 0:
                line.append(round(float(catAnalysis[d]['true']) /
                                  (catAnalysis[d]['true'] + catAnalysis[d]['false']) * 100, 0))
            else:
                line.append(0)
            line.append(round((catAnalysis[d]['true'] + catAnalysis[d]['false']) / allOccurrences * 100, 0))
        table.append(line)
    return table


def getErrorAnalysis(corpus):
    correctlyIdentified = {mweKey: {}, mwtKey: {}, freqSeenKey: {}, barelySeenKey: {}, partSeenKey: {}, newKey: {},
                           interleavingKey: {}, embeddedKey: {}}
    misidentified = {mweKey: {}, freqSeenKey: {}, barelySeenKey: {}, partSeenKey: {}, newKey: {}}
    nonIdentified = {mweKey: {}, mwtKey: {}, seenKey: {}, barelySeenKey: {}, freqSeenKey: {},
                     newKey: {}, partSeenKey: {},
                     interleavingKey: {}, embeddedKey: {}}
    testMWEs = {mweKey: {}, mwtKey: {}, interleavingKey: {}, embeddedKey: {}}
    for s in corpus.testingSents:
        s.recognizeEmbedded(annotated=False)
        mweIdxs = set([mwe.getTokenPositionString() for mwe in s.vMWEs])
        for mwe in s.identifiedVMWEs:
            if mwe.getTokenPositionString() in mweIdxs:
                updateCorrectlyIdentified(mwe, s, corpus.mweDictionary, corpus.mweTokenDictionary, correctlyIdentified)
            else:
                updateMisIdentified(mwe, s, misidentified, corpus.mweDictionary, corpus.mweTokenDictionary)

        idenMweIdxs = set([mwe.getTokenPositionString() for mwe in s.identifiedVMWEs])
        for mwe in s.vMWEs:
            updateTestInfo(mwe, testMWEs, s, corpus.mweDictionary)
            if mwe.getTokenPositionString() not in idenMweIdxs:
                updatenonIdentified(mwe, s, nonIdentified, corpus.mweDictionary, corpus.mweTokenDictionary)

    return {misIdentifiedKey: misidentified, corIdentifiedKey: correctlyIdentified,
            nonIdentifiedKey: nonIdentified, testSetKey: testMWEs, }


def getCatAnalysis(corpus):
    cats = {'LVC.full': {'true': 0, 'false': 0},
            'LVC.cause': {'true': 0, 'false': 0},
            'VID': {'true': 0, 'false': 0},
            'IRV': {'true': 0, 'false': 0},
            'VPC.full': {'true': 0, 'false': 0},
            'VPC.semi': {'true': 0, 'false': 0},
            'MVC': {'true': 0, 'false': 0},
            'IAV': {'true': 0, 'false': 0}}
    for s in corpus.testingSents:
        idenMweIdxs = set([mwe.getTokenPositionString() for mwe in s.identifiedVMWEs])
        for mwe in s.vMWEs:
            if mwe.type2 not in cats:
                continue
                # cats[mwe.type2] = {}
            if mwe.getTokenPositionString() in idenMweIdxs:
                cats[mwe.type2]['true'] = 1 if 'true' not in cats[mwe.type2] else 1 + cats[mwe.type2]['true']
            else:
                cats[mwe.type2]['false'] = 1 if 'false' not in cats[mwe.type2] else 1 + cats[mwe.type2]['false']
    return cats


def getDataSetStr():
    datasetConf = configuration['dataset']
    return 'ST2' if datasetConf['sharedtask2'] else \
        ('FTB' if datasetConf['ftb'] else ('DiMSUM' if datasetConf['dimsum'] else 'ST1'))


def getModeStr():
    modelConf = configuration['xp']
    return 'SVM' if modelConf['linear'] else (
        'KIIPER' if modelConf['kiperwasser'] else (
            'RNN' if modelConf['rnn'] else 'MLP'))


def reportStats(analysisDic, catAnalysis, folder):
    res = ''
    for d in sorted(analysisDic.keys()):
        res += '### %s\n\n' % d.upper()
        for s in sorted(analysisDic[d].keys()):
            res += '%s: number %d, occurrences: %d\n\n' % (
                s.upper(), len(analysisDic[d][s]), getOccurrences(analysisDic[d][s]))
    res += '### Categories : \n\n'
    for d in sorted(catAnalysis.keys()):
        res += d + '\t\t : %d / %d \n\n' % (catAnalysis[d]['true'], catAnalysis[d]['true'] + catAnalysis[d]['false'])
    with open(os.path.join(folder, 'stats.md'), 'w') as f:
        f.write(res)


def getErrorAnalysisFolder(lang):
    folder = os.path.join(configuration['path']['projectPath'],
                          configuration['path']['errorAnalysis'],
                          getDataSetStr(), getModeStr(), lang)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def report(analysisDic, catAnalysis, lang):
    folder = getErrorAnalysisFolder(lang)
    reportStats(analysisDic, catAnalysis, folder)
    for d in sorted(analysisDic.keys()):
        res, idx = '', 1
        if d != testSetKey:
            for s in sorted(analysisDic[d].keys()):
                if analysisDic[d][s] and s not in [mweKey]:
                    res += '## %s\n\n' % s.upper()
                    idx = 1
                    for k in sorted(analysisDic[d][s].keys()):
                        res += '%d. %s : %d  / %d :\n\n\t\t %s\n\n' % \
                               (idx, k, analysisDic[d][s][k][0], analysisDic[d][s][k][1], '\n\n\t\t'.
                                join(str(i) for i in analysisDic[d][s][k][2]))
                        idx += 1
            with open(os.path.join(folder, '%s.md' % d), 'w') as f:
                f.write(res)


def updateMisIdentified(mwe, sent, misidentified, mweDic, mweTokenDic):
    updateDict(misidentified[mweKey], mwe, sent, mweDic)
    if mwe.getLemmaString() in mweDic:
        if mweDic[mwe.getLemmaString()] > 5:
            updateDict(misidentified[freqSeenKey], mwe, sent, mweDic)
        else:
            updateDict(misidentified[barelySeenKey], mwe, sent, mweDic)
    else:
        partiallySeen = False
        for t in mwe.tokens:
            if t.lemma in mweTokenDic:
                updateDict(misidentified[partSeenKey], mwe, sent, mweDic, mweTokenDic)
                partiallySeen = True
                break
        if not partiallySeen:
            updateDict(misidentified[newKey], mwe, sent, mweDic)


def updateCorrectlyIdentified(mwe, sent, mweDic, mweTokenDic, correctlyIdentified):
    updateDict(correctlyIdentified[mweKey], mwe, sent, mweDic)
    if mwe.getLemmaString() in mweDic:
        if mweDic[mwe.getLemmaString()] > 5:
            updateDict(correctlyIdentified[freqSeenKey], mwe, sent, mweDic)
        else:
            updateDict(correctlyIdentified[barelySeenKey], mwe, sent, mweDic)
    else:
        partiallySeen = False
        for t in mwe.tokens:
            if t.lemma in mweTokenDic:
                updateDict(correctlyIdentified[partSeenKey], mwe, sent, mweDic, mweTokenDic)
                partiallySeen = True
                break
        if not partiallySeen:
            updateDict(correctlyIdentified[newKey], mwe, sent, mweDic)
    if mwe.isEmbedded:
        updateDict(correctlyIdentified[embeddedKey], mwe, sent, mweDic)
    if len(mwe.tokens) == 1:
        updateDict(correctlyIdentified[mwtKey], mwe, sent, mweDic)


def updatenonIdentified(mwe, sent, nonIdentified, mweDic, mweTokenDic):
    updateDict(nonIdentified[mweKey], mwe, sent, mweDic)
    if len(mwe.tokens) == 1:
        updateDict(nonIdentified[mwtKey], mwe, sent, mweDic)
    if mwe.isEmbedded:
        updateDict(nonIdentified[embeddedKey], mwe, sent, mweDic)
    if mwe.isInterleaving:
        updateDict(nonIdentified[interleavingKey], mwe, sent, mweDic)
    if mwe.getLemmaString() in mweDic:
        # updateDict(nonIdentified[seenKey], mwe, sent, mweDic)
        if mweDic[mwe.getLemmaString()] > 5:
            updateDict(nonIdentified[freqSeenKey], mwe, sent, mweDic)
        else:
            updateDict(nonIdentified[barelySeenKey], mwe, sent, mweDic)
        # nonIdentified[seenKey][mwe.getLemmaString()] = [mweDic[mwe.getLemmaString()], []]
    else:
        # updateDict(nonIdentified[newKey], mwe, sent, mweDic)
        isPartiallySeen = False
        for t in mwe.tokens:
            if t.getLemma() in mweTokenDic:
                updateDict(nonIdentified[partSeenKey], mwe, sent, mweDic, mweTokenDic)
                isPartiallySeen = True
                break
        if not isPartiallySeen:
            updateDict(nonIdentified[newKey], mwe, sent, mweDic, mweTokenDic)


def getTextZone(mwe, sent, window=3):
    pos = [t.position for t in mwe.tokens]
    minPos = min(pos)  # - 2 if min(pos) > 1 else min(pos) - 1
    idx = 0
    while minPos > 0 and idx <= window:
        minPos -= 1
        idx += 1
    maxPos = max(pos)
    idx = 0
    while maxPos < len(sent.tokens) - 1 and idx <= window:
        maxPos += 1
        idx += 1
    return ' '.join(sent.tokens[i].text for i in range(minPos, maxPos) if len(sent.tokens) > i >= 0)


def updateTestInfo(mwe, testMWEs, sent, trainMWEDic):
    updateDict(testMWEs[mweKey], mwe, sent, trainMWEDic)
    if len(mwe.tokens) == 1:
        updateDict(testMWEs[mwtKey], mwe, sent, trainMWEDic)
    if mwe.isEmbedded:
        updateDict(testMWEs[embeddedKey], mwe, sent, trainMWEDic)
    if mwe.isInterleaving:
        updateDict(testMWEs[interleavingKey], mwe, sent, trainMWEDic)


def getOccurrences(dic):
    return sum([dic[k][0] for k in dic]) if dic else 0


def updateDict(dic, mwe, sent, trainMWEDic, mWETokenDic=None):
    if mWETokenDic:
        mweString = ''
        for t in mwe.tokens:
            if t.getLemma() in mWETokenDic:
                mweString += '**%s** ' % (t.getLemma())
            else:
                mweString += t.getLemma() + ' '
        mweString = mweString[:-1]
    else:
        mweString = mwe.getLemmaString()
    textZone = getTextZone(mwe, sent)
    trainFreq = 0 if mwe.getLemmaString() not in trainMWEDic else trainMWEDic[mwe.getLemmaString()]
    if mweString not in dic:
        dic[mweString] = [1, trainFreq, [textZone]]
    else:
        dic[mweString][0] = dic[mweString][0] + 1
        dic[mweString][1] = trainFreq
        dic[mweString][2].append(textZone)


def readIdentifiedCorpus(lang, dataset, division, testFileName):
    setTrainAndTest(division)
    setDataSet(dataset)
    testFileName = getOutputFile(testFileName)
    corpus = Corpus(lang)
    if dataset == Dataset.dimsum:
        testSents = readDiMSUM(testFileName)
    elif dataset == Dataset.ftb:
        testSents = readFTB(testFileName)
    else:
        testSents = readCuptFile(testFileName)
    for i, s in enumerate(corpus.testingSents):
        if i < len(testSents):
            s.identifiedVMWEs = testSents[i].vMWEs
        else:
            sys.stderr.write(
                'Test languages are not equal {0} {1} {2}'.format(lang, len(testSents), len(corpus.testingSents)))
    return corpus


def evaluateFile(lang, xpMode, dataset, division, testFileName):
    setXPMode(xpMode)
    setDataSet(dataset)
    setTrainAndTest(division)
    corpus = Corpus(lang)
    testSents = readCuptFile(testFileName)
    for i, s in enumerate(corpus.testingSents):
        testSents[i].identifiedVMWEs = testSents[i].vMWEs
        testSents[i].vMWEs = s.vMWEs
    return evaluate(testSents)[1]


def getTokenBasedMcNemarScore(args, goldenFile, sys1File, sys2File):
    goldSents = readCuptFile(goldenFile)
    sys1Sents = readCuptFile(sys1File)
    sys2Sents = readCuptFile(sys2File)
    tt, ff, tf, ft = 0, 0, 0, 0
    for i in range(len(goldSents)):
        if len(goldSents[i].vMWEs) > 1:
            pass
        gLbls = ['*' if t.parentMWEs else '-' for t in goldSents[i].tokens]
        # gLbls = ['.'.join(str(v.getTokenPositionString()) for v in t.parentMWEs)
        #  if t.parentMWEs else '-' for t in goldSents[i].tokens]
        sys1Lbls = ['*' if t.parentMWEs else '-' for t in sys1Sents[i].tokens]
        # sys1Lbls = ['.'.join(str(v.getTokenPositionString()) for v in t.parentMWEs)
        # if t.parentMWEs else '-' for t in sys1Sents[i].tokens]
        sys2Lbls = ['*' if t.parentMWEs else '-' for t in sys2Sents[i].tokens]
        # sys2Lbls = ['.'.join(str(v.getTokenPositionString()) for v in t.parentMWEs)
        # if t.parentMWEs else '-' for t in sys2Sents[i].tokens]
        for j in range(len(gLbls)):
            if sys1Lbls[j] == gLbls[j] and sys1Lbls[j] != sys2Lbls[j]:
                tf += 1
            elif sys2Lbls[j] == gLbls[j] and sys1Lbls[j] != sys2Lbls[j]:
                ft += 1
            elif sys1Lbls[j] == sys2Lbls[j] == gLbls[j]:
                tt += 1
            else:
                ff += 1
    mc = mcnemar([[tt, tf], [ft, ff]], exact=False, correction=True)
    hypothesis = 'fail to reject H0' if mc.pvalue > .05 else 'reject H0'
    return ','.join(str(t) for t in [args[0], args[1], args[2], tt, tf, ft, ff, mc.pvalue, hypothesis])


def getMWEBasedcNemarScore(args, goldenFile, sys1File, sys2File):
    goldSents = readCuptFile(goldenFile)
    sys1Sents = readCuptFile(sys1File)
    sys2Sents = readCuptFile(sys2File)
    tt, ff, tf, ft, mc = 0, 0, 0, 0, None
    for i in range(len(goldSents)):
        goldenMWEs, sys1MWEs, sys2MWEs = set(), set(), set()
        for v in goldSents[i].vMWEs:
            goldenMWEs.add(v.getTokenPositionString())
        for v in sys1Sents[i].vMWEs:
            sys1MWEs.add(v.getTokenPositionString())
        for v in sys2Sents[i].vMWEs:
            sys2MWEs.add(v.getTokenPositionString())
        for v in sys1Sents[i].vMWEs:
            k = v.getTokenPositionString()
            if k not in sys2MWEs:
                tf += 1
            else:
                tt += 1
        for v in sys2Sents[i].vMWEs:
            k = v.getTokenPositionString()
            if k not in sys1MWEs:
                ft += 1
        for v in goldSents[i].vMWEs:
            k = v.getTokenPositionString()
            if k not in sys1MWEs and k not in sys2MWEs:
                ff += 1
    mc = mcnemar([[tt, tf], [ft, ff]], exact=False, correction=True)
    hypothesis = 'reject H0' if mc.pvalue < 0.05 else 'fail to reject H0'
    return ','.join(str(t) for t in [args[0], args[1], args[2], tt, tf, ft, ff, mc.pvalue, hypothesis])


def renameFolders():
    import os
    folders = ['/Users/halsaied/PycharmProjects/NNIdenSys/Results/ST2/Linear',
               '/Users/halsaied/PycharmProjects/NNIdenSys/Results/ST2/MLP']
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                os.rename(os.path.join(folder, filename), os.path.join(folder, filename[3:]))


def getMecNamers(division, tokenBased=False, ):
    divMaj = 'FixedSize' if division == Evaluation.fixedSize else 'Corpus'
    divMin = 'fixedsize' if division == Evaluation.fixedSize else 'corpus'
    outPutFolder = '/Users/halsaied/PycharmProjects/NNIdenSys/Results/Output/'
    goldTestFilePattern = outPutFolder + 'ST2/Linear/{0}/12.11.{1}.{2}.gold.cupt'
    sys1FilePattern = outPutFolder + '{0}/{1}/{2}/12.11.{3}.{4}.cupt'
    sys2FilePattern = outPutFolder + '{0}/{1}/{2}/12.11.{3}.{4}.cupt'
    from xpTools import allSharedtask2Lang
    res = ''
    for l in allSharedtask2Lang:
        goldTestFile = goldTestFilePattern.format(divMaj, divMin, l)
        sys1File = sys1FilePattern.format('ST2', 'Linear',divMaj, divMin, l)
        linScore = evaluateFile(l, XpMode.linear, Dataset.sharedtask2,division, sys1File)
        sys2File = sys2FilePattern.format('ST2', 'MLP',divMaj, divMin, l)
        mlpScore = evaluateFile(l, None, Dataset.sharedtask2,division, sys2File)
        res += getTokenBasedMcNemarScore([l, linScore, mlpScore], goldTestFile, sys1File, sys2File) if tokenBased else \
            getMWEBasedcNemarScore([l, linScore, mlpScore], goldTestFile, sys1File, sys2File)
        print res
    print res


def getOutputFile(fileName):
    datasetConf, modelConf = configuration['dataset'], configuration['xp']
    dataset = 'ST2' if datasetConf['sharedtask2'] else \
        ('FTB' if datasetConf['ftb'] else ('DiMSUM' if datasetConf['dimsum'] else 'ST1'))
    model = 'Linear' if modelConf['linear'] else (
        'Kiperwasser' if modelConf['kiperwasser'] else (
            'RNN' if modelConf['rnn'] else 'MLP'))
    devision = 'FixedSize' if configuration['evaluation']['fixedSize'] else 'Corpus'
    return os.path.join(configuration['path']['projectPath'],
                        configuration['path']['output'],
                        dataset, model, devision, fileName)  # today + lang.upper() + '.cupt')


if __name__ == '__main__':
    getMecNamers(Evaluation.corpus)
    # errorAnalysis('FR', XpMode.linear, '../Results/ST2/MLP/FR.txt')
    # analyzeCorpus('FR', XpMode.linear, '../Results/ST2/MLP/FR.txt')
    # csvFile = ''

    # analyzeCorpus\

    # getCatAnalysisTable(xpMode=XpMode.linear,
    #                        dataset=Dataset.sharedtask2,
    #                       division=Evaluation.fixedSize)

    # res = '\n'
    # pathh = '/Users/halsaied/PycharmProjects/NNIdenSys/Results/ST2/Linear'
    # for f in os.listdir(pathh):
    #     res += analyzeCorpus(f[:2], xpMode=XpMode.linear, dataset=Dataset.sharedtask2, testFileName=pathh + '/' + f)
    # print res
    # print analyzeCorpus('EN', xpMode=XpMode.linear, dataset=Dataset.dimsum,
    #                     testFileName='/Users/halsaied/PycharmProjects/NNIdenSys/Results/FTB/Linear/EN.dimsum')
    # print analyzeCorpus('FR', xpMode=XpMode.linear, dataset=Dataset.ftb,
    #                     testFileName='/Users/halsaied/PycharmProjects/NNIdenSys/Results/FTB/Linear/FR.ftb')
