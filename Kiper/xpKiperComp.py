import logging
import random

import torch

from corpus import *
from identification import xp

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)

    random.seed(0)
    torch.manual_seed(0)

    configuration['kiperwasser'].update({
        'focusedElemNum': 8,
        'wordDim': 100,
        'posDim': 30,
        'denseActivation': 'tanh',
        'dense1': 30,
        'dense2': 0,
        'denseDropout': False,
        'optimizer': 'adagrad',
        'lr': 0.07,
        'epochs': 30,
        'batch': 1,
        'lstmDropout': .2,
        'lstmLayerNum': 1,
        'lstmUnitNum': 60,
        'verbose': False,
        'moreTrans': True
    })

    configuration['sampling']['importantSentences'] = True
    configuration['dataset']['sharedtask2'] = True
    configuration['xp']['kiperwasser'] = True
    configuration['evaluation']['fixedSize'] = True
    configuration['evaluation']['debugTrainNum'] = 200
    configuration['model']['embedding']['compactVocab'] = True
    configuration['model']['embedding']['lemma'] = False

    from identification import createRSGGrid, runRSGThread

    filename = 'compactKiperGrid.p'
    langs = ['BG' ]  # , 'PT', 'TR']
    createRSGGrid(filename=filename)
    runRSGThread(langs, xpNum=200, compact=True, filename=filename)
    # xp(['FR'], xpNum=1, compact=True)
