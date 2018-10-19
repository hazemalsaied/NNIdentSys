from statsmodels.stats.contingency_tables import mcnemar

from corpus import readCuptFile, Corpus
from evaluation import evaluate
from xpTools import setXPMode, setDataSet, Evaluation, setTrainAndTest


def errorAnalysis(lang, testFileName, dataset, xpMode):
    setXPMode(xpMode)
    setDataSet(dataset)
    setTrainAndTest(Evaluation.corpus)
    corpus = Corpus(lang)
    testSents = readCuptFile(testFileName)
    for i, s in enumerate(corpus.testingSents):
        testSents.identifiedVMWEs = testSents.vMWEs
        testSents.vMWEs = s.vMWEs
    evaluate(testSents)

    # TODO
    pass


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
    getMecNamers(tokenBased=False)
