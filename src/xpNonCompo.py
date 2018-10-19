from xpTools import *


def setOptimalRSG():
    samling = configuration['sampling']
    samling['importantSentences'] = True
    samling['overSampling'] = True
    samling['sampleWeight'] = True
    samling['favorisationCoeff'] = 6
    samling['focused'] = True

    configuration['mlp']['optimizer'] = 'adagrad'
    configuration['mlp']['lr'] = 0.059

    configuration['mlp']['lemma'] = True
    configuration['mlp']['posEmb'] = 42
    configuration['mlp']['tokenEmb'] = 480
    configuration['mlp']['compactVocab'] = False

    configuration['mlp']['dense1'] = True
    configuration['mlp']['dense1UnitNumber'] = 58
    configuration['mlp']['dense1Dropout'] = 0.429


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    setOptimalRSG()
    langs = ['FR']
    # xp(langs, Dataset.sharedtask2, None, None)#, mlpInLinear=True)  # linearInMlp=True)
    xp(['FR'], Dataset.ftb, None, Evaluation.corpus)  # , linearInMlp=True)  # linearInMlp=True)
    configuration['mlp']['compactVocab'] = True
    xp(['FR'], Dataset.ftb, None, Evaluation.corpus)  # , linearInMlp=True)  # linearInMlp=True)
    configuration['mlp']['compactVocab'] = False
    configuration['others']['removeFtbMWT'] = True
    xp(['FR'], Dataset.ftb, None, Evaluation.corpus)  # , linearInMlp=True)  # linearInMlp=True)
    configuration['mlp']['compactVocab'] = True
    xp(['FR'], Dataset.ftb, None, Evaluation.corpus)  # , linearInMlp=True)  # linearInMlp=True)
