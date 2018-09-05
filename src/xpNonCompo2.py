import sys

from config import configuration
from identification import xp

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    configuration['xp']['kiperwasser'] = True

    configuration['kiperwasser'] = {
        'wordDim': 100,
        'posDim': 25,
        'layerNum': 2,
        'activation': 'tanh',
        'optimizer': 'adam',
        'lr': 0.1,
        'dropout': .3,
        'epochs': 20,
        'batch': 1,
        'dense1': 100,
        'dense2': 0,
        'denseDropout': False,
        'lstmDropout': 0.25,
        'lstmLayerNum': 2,
        'focusedElemNum': 8,
        'lstmUnitNum': 125
    }
    configuration["sampling"]["importantSentences"] = True
    # configuration["evaluation"]["fixedSize"] = True
    configuration["evaluation"]["debugTrainNum"] = 20

    sys.stdout.write(str(configuration['kiperwasser']))

    xp(['FR'])
