import sys

from xpNonCompo import xp

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    from config import configuration
    from xpNonCompo import setOptimalRandomGridParameters, allSharedtask2Lang

    configuration['model']['train']['earlyStop'] = True
    configuration['dataset']['sharedtask2'] = True
    setOptimalRandomGridParameters()
    configuration['evaluation']['trainVsTest'] = True
    xp(allSharedtask2Lang)
    # configuration['xp']['rnnNonCompo'] = True  # ['rnnNonCompo'] = True
    # configuration['evaluation']['debugTrainNum'] = 2000
    # xp(['FR'], train=False)
