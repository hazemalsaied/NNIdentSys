import logging

from xpTools import *
from config import configuration
if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)

    random.seed(0)
    torch.manual_seed(0)

    configuration['kiperwasser'].update({
        'focusedElemNum': 8,
        'wordDim': 89,
        'posDim': 30,
        'denseActivation': 'tanh',
        'dense1': 30,
        'dense2': 0,
        'denseDropout': False,
        'optimizer': 'adagrad',
        'lr': 0.07,
        'epochs': 5,
        'batch': 1,
        'rnnDropout': .2,
        'rnnLayerNum': 1,
        'rnnUnitNum': 60,
        'verbose': False,
        'moreTrans': True
    })

    configuration['sampling']['importantSentences'] = True
    configuration['evaluation']['debugTrainNum'] = 200
    configuration['mlp']['compactVocab'] = True
    configuration['mlp']['lemma'] = False

    filename = 'compactKiperGrid.p'
    configuration['kiperwasser']['eager'] = True
    configuration['kiperwasser']['gru'] = False

    from rsg import createRSG
    createRSG('kiper.p',XpMode.kiperwasser)
    configuration['kiperwasser']['sampling'] = True
    xp(['BG'], Dataset.sharedtask2, XpMode.kiperwasser, Evaluation.fixedSize)
    # from rsg import createRSGGrid, runRSGThread
    # createRSGGrid(filename=filename)
    # runRSGThread(langs, xpNum=150, compact=True, filename=filename)
    # xp(['FR'], xpNum=1, compact=True)
