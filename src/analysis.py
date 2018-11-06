from statsmodels.stats.contingency_tables import mcnemar

from evaluation import evaluate
from xpTools import *


def analyzeCorpus(lang, xpMode, dataset, testFileName):
    corpus = readIdentifiedCorpus(lang, dataset, testFileName)
    evaluate(corpus.testingSents)
    analysis = errorAnalysis(corpus)
    exportEnalysis(analysis, lang, xpMode, dataset)
    totalOcc = getOccurrences(analysis['4.Test set']['mwe'])

    seenLessThan5Times = 0
    seenAndNonIdentifiedPercentage = int(
        float(getOccurrences(analysis['3.Non identified MWEs']['seen'])) / totalOcc * 100)
    for k in analysis['3.Non identified MWEs']['seen']:
        if analysis['3.Non identified MWEs']['seen'][k][0] <= 5:
            seenLessThan5Times += 1

    seenLessThan5TimePercentageInSeen = int(float(seenLessThan5Times) / len(analysis['3.Non identified MWEs']['seen']) *100)

    newAndNonIdentifiedPercentage = int(
        float(getOccurrences(analysis['3.Non identified MWEs']['new'])) / totalOcc * 100)
    return '\n{0},{1},{2},{3}'.format(
        lang,
        seenAndNonIdentifiedPercentage,
        seenLessThan5TimePercentageInSeen,
        newAndNonIdentifiedPercentage
    )
    # res = lang + ','
    # for d in sorted(analysis.keys(), reverse=True):
    #     if d not in ['2.Correctly identified MWEs', '4.Test set']:  # print d
    #         for s in sorted(analysis[d].keys()):
    #             if s not in ['mwt', 'embedded', 'interleaving']:
    #                 # res += s + ' num,occ, percentage,'
    #                 occ = getOccurrences(analysis[d][s])
    #                 res += str(len(analysis[d][s])) + ',' + str(occ) + ',' + (str(
    #                     round(100 * float(len(analysis[d][s])) / float(totalOcc), 1)) + ',' if occ else '0,')
    # return '\n' + res[:-1]


def exportEnalysis(analysis, lang, dataset, xpMode):
    filePath = '../Results/ErrorAnalysis/{0}.{1}.{2}.md'.format(dataset, lang,
                                                                str(xpMode).split('.')[-1]) if xpMode else 'MLP'
    with open(filePath, 'w') as f:
        f.write(report(analysis))
        sys.stdout.write(filePath.split('/')[-1] + ' has been created in Results/ErrorAnalysis folder!')


def errorAnalysis(corpus):
    correctlyIdentified = {'mwe': {}, 'mwt': {}, 'seen': {}, 'new': {}, 'interleaving': {}, 'embedded': {}}
    misidentified = {'mwe': {}, 'partially_Seen': {}}
    nonIdentified = {'mwe': {}, 'mwt': {}, 'seen': {}, 'new': {}, 'interleaving': {}, 'embedded': {}}
    testMWEs = {'mwe': {}, 'mwt': {}, 'interleaving': {}, 'embedded': {}}
    for s in corpus.testingSents:
        s.recognizeEmbedded(annotated=False)
        mweIdxs = set([mwe.getTokenPositionString() for mwe in s.vMWEs])
        for mwe in s.identifiedVMWEs:
            if mwe.getTokenPositionString() in mweIdxs:
                updateCorrectlyIdentified(mwe, s, corpus.mweDictionary, correctlyIdentified)
            else:
                updateMisIdentified(mwe, s, corpus.mweTokenDictionary, misidentified)

        idenMweIdxs = set([mwe.getTokenPositionString() for mwe in s.identifiedVMWEs])
        for mwe in s.vMWEs:
            updateTestInfo(mwe, testMWEs, s)
            if mwe.getTokenPositionString() not in idenMweIdxs:
                updatenonIdentified(mwe, s, corpus.mweDictionary, nonIdentified)

    return {'1.Misidentified MWEs': misidentified, '2.Correctly identified MWEs': correctlyIdentified,
            '3.Non identified MWEs': nonIdentified, '4.Test set': testMWEs, }


def updateMisIdentified(mwe, sent, mweTokenDic, misidentified):
    updateDict(misidentified['mwe'], mwe, sent)
    for t in mwe.tokens:
        if t.lemma in mweTokenDic:
            updateDict(misidentified['partially_Seen'], mwe, sent)
            break


def updateCorrectlyIdentified(mwe, sent, mweDic, correctlyIdentified):
    updateDict(correctlyIdentified['mwe'], mwe, sent)
    if mwe.getLemmaString() in mweDic:
        updateDict(correctlyIdentified['seen'], mwe, sent)
    else:
        updateDict(correctlyIdentified['new'], mwe, sent)
    if mwe.isEmbedded:
        updateDict(correctlyIdentified['embedded'], mwe, sent)
    if len(mwe.tokens) == 1:
        updateDict(correctlyIdentified['mwt'], mwe, sent)


def updatenonIdentified(mwe, sent, mweDic, nonIdentified):
    updateDict(nonIdentified['mwe'], mwe, sent)
    if len(mwe.tokens) == 1:
        updateDict(nonIdentified['mwt'], mwe, sent)
    if mwe.isEmbedded:
        updateDict(nonIdentified['embedded'], mwe, sent)
    if mwe.isInterleaving:
        updateDict(nonIdentified['interleaving'], mwe, sent)
    if mwe.getLemmaString() in mweDic:
        nonIdentified['seen'][mwe.getLemmaString()] = [mweDic[mwe.getLemmaString()], []]
    else:
        updateDict(nonIdentified['new'], mwe, sent)


def getTextZone(mwe, sent):
    pos = [t.position for t in mwe.tokens]
    minPos = min(pos) - 2 if min(pos) > 1 else min(pos) - 1
    maxPos = max(pos) if max(pos) == len(sent.tokens) else max(pos) + 1
    return ' '.join(sent.tokens[i].text for i in range(minPos, maxPos) if len(sent.tokens) > i >= 0)


def updateTestInfo(mwe, testMWEs, sent):
    updateDict(testMWEs['mwe'], mwe, sent)
    if len(mwe.tokens) == 1:
        updateDict(testMWEs['mwt'], mwe, sent)
    if mwe.isEmbedded:
        updateDict(testMWEs['embedded'], mwe, sent)
    if mwe.isInterleaving:
        updateDict(testMWEs['interleaving'], mwe, sent)


def getOccurrences(dic):
    if dic:
        return sum([dic[k][0] for k in dic])
    return 0


def report(analysisDic):
    res = ''
    for d in sorted(analysisDic.keys()):
        res += '### %s\n\n' % d.upper()
        for s in sorted(analysisDic[d].keys()):
            res += '%s: number %d, occurrences: %d\n\n' % (
                s.upper(), len(analysisDic[d][s]), getOccurrences(analysisDic[d][s]))
    sys.stdout.write(res)
    for d in sorted(analysisDic.keys()):
        res += '# %s\n\n' % d.upper()
        for s in sorted(analysisDic[d].keys()):
            res += '## %s\n\n' % s.upper()
            if analysisDic[d][s]:
                for k in sorted(analysisDic[d][s].keys()):
                    res += '%s : %d : %s\n\n' % (k, analysisDic[d][s][k][0],
                                                 ':'.join(str(i) for i in analysisDic[d][s][k][1]))
    return res


def updateDict(dic, mwe, sent):
    mweString = mwe.getLemmaString()
    textZone = getTextZone(mwe, sent)
    if mweString not in dic:

        dic[mweString] = [1, [textZone]]
    else:
        dic[mweString][0] = dic[mweString][0] + 1
        dic[mweString][1].append(textZone)


def readIdentifiedCorpus(lang, dataset, testFileName):
    setTrainAndTest(Evaluation.corpus)
    setDataSet(dataset)
    corpus = Corpus(lang)
    if dataset == Dataset.dimsum:
        testSents = readDiMSUM(testFileName)
    elif dataset == Dataset.dimsum:
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


def evaluateFile(lang, xpMode, dataset, testFileName):
    setXPMode(xpMode)
    setDataSet(dataset)
    setTrainAndTest(Evaluation.corpus)
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


def getMecNamers(tokenBased=False):
    goldTestFilePattern = '/Users/halsaied/PycharmProjects/NNIdenSys/Results/Gold/{0}/test.cupt'
    sys1FilePattern = '/Users/halsaied/PycharmProjects/NNIdenSys/Results/{0}/{1}/{2}.txt'
    sys2FilePattern = '/Users/halsaied/PycharmProjects/NNIdenSys/Results/{0}/{1}/{2}.txt'
    from xpTools import allSharedtask2Lang
    for l in allSharedtask2Lang:
        goldTestFile = goldTestFilePattern.format(l)
        sys1File = sys1FilePattern.format('ST2', 'Linear', l)
        linScore = 0  # evaluateFile(l, XpMode.linear, Dataset.sharedtask2, sys1File)
        sys2File = sys2FilePattern.format('ST2', 'MLP', l)
        mlpScore = 0  # evaluateFile(l, XpMode.linear, Dataset.sharedtask2, sys2File)
        print getTokenBasedMcNemarScore([l, linScore, mlpScore], goldTestFile, sys1File, sys2File) if tokenBased else \
            getMWEBasedcNemarScore([l, linScore, mlpScore], goldTestFile, sys1File, sys2File)


if __name__ == '__main__':
    # getMecNamers(tokenBased=False)
    # errorAnalysis('FR', XpMode.linear, '../Results/ST2/MLP/FR.txt')
    # analyzeCorpus('FR', XpMode.linear, '../Results/ST2/MLP/FR.txt')
    # csvFile = ''
    res = '\n'
    pathh = '/Users/halsaied/PycharmProjects/NNIdenSys/Results/ST2/Linear'
    for f in os.listdir(pathh):
        res += analyzeCorpus(f[:2], xpMode=XpMode.linear, dataset=Dataset.sharedtask2, testFileName=pathh + '/' + f)
    print res
    print analyzeCorpus('EN', xpMode=XpMode.linear, dataset=Dataset.dimsum,
                      testFileName='/Users/halsaied/PycharmProjects/NNIdenSys/Results/FTB/Linear/EN.dimsum')
    print analyzeCorpus('FR', xpMode=XpMode.linear, dataset=Dataset.ftb,
                        testFileName='/Users/halsaied/PycharmProjects/NNIdenSys/Results/FTB/Linear/FR.ftb')
