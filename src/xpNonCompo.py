import sys

from config import *
from identification import xp

langs = ['FR']

allSharedtask1Lang = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT',
                      'LT', 'MT', 'PL', 'PT', 'RO', 'SV', 'SL', 'TR']

allSharedtask2Lang = ['BG', 'DE', 'EL', 'EN', 'ES', 'EU', 'FA', 'FR', 'HE', 'HI',
                      'HR', 'HU', 'IT', 'LT', 'PL', 'PT', 'RO', 'SL', 'TR']

pilotLangs = ['BG', 'PT', 'TR']


def setOptimalRandomGridParameters():
    samling = configuration['sampling']
    samling['importantSentences'] = True
    samling['overSampling'] = True
    samling['sampleWeight'] = True
    samling['favorisationCoeff'] = 6
    samling['focused'] = True

    configuration['model']['train']['optimizer'] = 'adagrad'
    configuration['model']['train']['lr'] = 0.059

    embConf = configuration['model']['embedding']
    embConf['active'] = True
    embConf['usePos'] = True
    embConf['lemma'] = True
    embConf['posEmb'] = 42
    embConf['tokenEmb'] = 480
    embConf['compactVocab'] = False

    dense1Conf = configuration['model']['mlp']['dense1']
    dense1Conf['active'] = True
    dense1Conf['unitNumber'] = 58
    dense1Conf['dropout'] = 0.429


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    from identification import setDataSet, setTrainAndTest

    setDataSet(Dataset.sharedtask2)
    setOptimalRandomGridParameters()

    setTrainAndTest(Evaluation.corpus)
    xp(allSharedtask2Lang, xpNum=1)
    setTrainAndTest(Evaluation.trainVsTest)
    xp(allSharedtask2Lang, xpNum=1)
    setTrainAndTest(Evaluation.trainVsDev)
    xp(allSharedtask2Lang, xpNum=1)
