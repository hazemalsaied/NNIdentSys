import sys

from config import *
from identification import xp

langs = ['FR']

allSharedtask1Lang = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT',
                      'LT', 'MT', 'PL', 'PT', 'RO', 'SV', 'SL', 'TR']

allSharedtask2Lang = ['BG', 'DE', 'EL', 'EN', 'ES', 'EU', 'FA', 'FR', 'HE', 'HI',
                      'HR', 'HU', 'IT', 'LT', 'PL', 'PT', 'RO', 'SL', 'TR']

pilotLangs = ['BG', 'PT', 'TR']


def setOptimalParameters():
    samling = configuration['sampling']
    samling['importantSentences'] = True
    samling['overSampling'] = True
    samling['sampleWeight'] = True
    samling['favorisationCoeff'] = 10

    configuration['model']['train']['optimizer'] = 'adagrad'
    configuration['model']['train']['lr'] = 0.02

    embConf = configuration['model']['embedding']
    embConf['active'] = True
    embConf['usePos'] = True
    embConf['lemma'] = True
    embConf['posEmb'] = 15
    embConf['tokenEmb'] = 200
    setDense1Conf(unitNumber=25)


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


def FTB(useAdam=False, epochs=40, focusedSampling=False, compactVocab=False, sampling=True):
    configuration['dataset']['FTB'] = True
    setOptimalParameters()
    if useAdam:
        configuration['model']['train']['optimizer'] = 'adam'
        configuration['model']['train']['lr'] = 0.01
    if not sampling:
        samling = configuration['sampling']
        samling['overSampling'] = False
        samling['sampleWeight'] = False

    if focusedSampling:
        configuration['sampling']['focused'] = False
        configuration['sampling']['mweRepeition'] = 40
    if compactVocab:
        configuration['model']['embedding']['compactVocab'] = True
    configuration['model']['train']['epochs'] = epochs
    xp(['FR'])
    # for tokEmb in [50, 100, 200]:
    #     configuration['model']['embedding']['tokenEmb'] = tokEmb
    #     xp(['FR'], train=True)
    #
    # configuration['model']['embedding']['tokenEmb'] = 50
    #
    # samling = configuration['sampling']
    # samling['sampleWeight'] = False
    # xp(['FR'], train=True)
    # samling['overSampling'] = False
    # xp(['FR'], train=True)


def exploreFTB():
    configuration['dataset']['FTB'] = True
    configuration['evaluation']['corpus'] = True
    setOptimalRandomGridParameters()
    xp(['FR'], xpNum=1)
    configuration['evaluation']['corpus'] = False
    configuration['evaluation']['trainVsTest'] = True
    xp(['FR'], xpNum=1)
    configuration['evaluation']['trainVsTest'] = False
    configuration['evaluation']['trainVsDev'] = True
    xp(['FR'], xpNum=1)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    from identification import setDataSet, setTrainAndTest

    setDataSet(Dataset.sharedtask2)
    setTrainAndTest(Evaluation.corpus)

    setOptimalRandomGridParameters()
    xp(allSharedtask2Lang, xpNum=1)
