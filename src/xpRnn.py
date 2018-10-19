import sys


def exploreBestConfs(langs):
    configuration['rnn']['gru'] = True
    configuration['rnn']['useDense'] = True
    configuration['mlp']['compactVocab'] = True

    bestConfs = [[283, 25, 94, 0.1, 64, 37, 0, 64],
                 [267, 78, 96, 0.3, 31, 40, 0, 16],
                 [410, 54, 51, 0.1, 25, 15, 0.1, 16]]
    for c in bestConfs:
        configuration['rnn']['wordDim'] = c[0]
        configuration['rnn']['posDim'] = c[1]
        configuration['rnn']['denseUnitNum'] = c[2]
        configuration['rnn']['denseDropout'] = c[3]
        configuration['rnn']['wordRnnUnitNum'] = c[4]
        configuration['rnn']['posRnnUnitNum'] = c[5]
        configuration['rnn']['rnnDropout'] = c[6]
        configuration['rnn']['batchSize'] = c[7]
        xp(langs, Dataset.sharedtask2, XpMode.rnn, Evaluation.fixedSize)


def setOptimalRSG():
    configuration['rnn']['gru'] = True
    configuration['rnn']['useDense'] = True
    configuration['mlp']['compactVocab'] = True
    configuration['mlp']['lemma'] = True

    configuration['sampling']['importantSentences'] = True
    configuration['sampling']['overSampling'] = True

    configuration['rnn']['wordDim'] = 410
    configuration['rnn']['posDim'] = 54
    configuration['rnn']['denseUnitNum'] = 51
    configuration['rnn']['denseDropout'] = 0.1
    configuration['rnn']['wordRnnUnitNum'] = 25
    configuration['rnn']['posRnnUnitNum'] = 15
    configuration['rnn']['rnnDropout'] = 0.1
    configuration['rnn']['batchSize'] = 16


def exploreExtraSampling(langs, dataset, xpMode, division):
    configuration['rnn']['gru'] = True
    configuration['rnn']['useDense'] = True
    configuration['mlp']['compactVocab'] = True

    bestConfs = [[410, 54, 51, 0.1, 25, 15, 0.1, 16]]
    # [283, 25, 94, 0.1, 64, 37, 0, 64],
    # [267, 78, 96, 0.3, 31, 40, 0, 16]]
    for c in bestConfs:

        configuration['rnn']['wordDim'] = c[0]
        configuration['rnn']['posDim'] = c[1]
        configuration['rnn']['denseUnitNum'] = c[2]
        configuration['rnn']['denseDropout'] = c[3]
        configuration['rnn']['wordRnnUnitNum'] = c[4]
        configuration['rnn']['posRnnUnitNum'] = c[5]
        configuration['rnn']['rnnDropout'] = c[6]
        configuration['rnn']['batchSize'] = c[7]
        xp(langs, dataset, xpMode, division)
        configuration['sampling']['focused'] = True
        xp(langs, dataset, xpMode, division)
        configuration['sampling']['sampleWeight'] = True
        for fc in [5, 15, 25]:
            configuration['sampling']['favorisationCoeff'] = fc
            xp(langs, dataset, xpMode, division)
        configuration['sampling']['sampleWeight'] = False
        configuration['sampling']['focused'] = False


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    from xpTools import *

    setOptimalRSG()
    configuration['sampling']['focused'] = True
    configuration['sampling']['sampleWeight'] = True
    configuration['sampling']['favorisationCoeff'] = 6
    configuration['rnn']['shuffle'] = True
    configuration['sampling']['importantSentences'] = True
    configuration['sampling']['overSampling'] = True

    from rsg import runRSG
    runRSG(allSharedtask2Lang, Dataset.sharedtask2, XpMode.rnn, Evaluation.fixedSize, 'rnn.p', xpNumByThread=80)
    # xp(allSharedtask2Lang, Dataset.sharedtask2, XpMode.rnn, Evaluation.corpus)

    # langs = ['FR']
    # xpMode = XpMode.rnn
    # dataset = Dataset.ftb
    # xp(langs, dataset, xpMode, Evaluation.corpus)
    # xp(langs, dataset, xpMode, Evaluation.trainVsTest)
    # xp(langs, dataset, xpMode, Evaluation.trainVsDev)
    #
    # langs = ['EN']
    # dataset = Dataset.dimsum
    # xp(langs, dataset, xpMode, Evaluation.corpus)
    # xp(langs, dataset, xpMode, Evaluation.trainVsTest)
