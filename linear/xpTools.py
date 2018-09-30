from enum import Enum
from theano import function, config, shared, tensor

import reports
from corpus import *
from identification import identify, crossValidation

langs = ['FR']

allSharedtask1Lang = ['BG', 'CS', 'DE', 'EL', 'ES', 'FA', 'FR', 'HE', 'HU', 'IT',
                      'LT', 'MT', 'PL', 'PT', 'RO', 'SV', 'SL', 'TR']

allSharedtask2Lang = ['BG', 'DE', 'EL', 'EN', 'ES', 'EU', 'FA', 'FR', 'HE', 'HI',
                      'HR', 'HU', 'IT', 'LT', 'PL', 'PT', 'RO', 'SL', 'TR']

pilotLangs = ['BG', 'PT', 'TR']
class Dataset(Enum):
    sharedtask2 = 0
    FTB = 1
    dimsum = 2


class Evaluation(Enum):
    cv = 0
    corpus = 1
    fixedSize = 2
    dev = 3
    trainVsDev = 4
    trainVsTest = 5


def xp(langs, xpNum=3, title='', seed=0):
    verifyGPU()
    evlaConf = configuration['evaluation']
    numpy.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    reports.createHeader(title)
    if not evlaConf['cv']['active']:
        for lang in langs:
            for i in range(xpNum):
                identify(lang)
    else:
        sys.stdout.write(reports.doubleSep + reports.tabs + 'CV Mode' + reports.doubleSep)
        for i in range(xpNum):
            crossValidation(langs)


def verifyGPU():
    vlen = 10 * 30 * 768
    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], tensor.exp(x))
    if numpy.any([isinstance(x.op, tensor.Elemwise) and
                  ('Gpu' not in type(x.op).__name__)
                  for x in f.maker.fgraph.toposort()]):
        sys.stdout.write(tabs + 'Attention: CPU used\n')
    else:
        sys.stdout.write(tabs + 'GPU Enabled')


class XpMode(Enum):
    linear = 0
    compo = 1
    pytorch = 2
    kiperwasser = 3
    rnn = 4
    rnnNonCompo = 5


def setTrainAndTest(v):
    configuration['evaluation'].update({
        'cv': {'active': True if v == Evaluation.cv else False},
        'corpus': True if v == Evaluation.corpus else False,
        'fixedSize': True if v == Evaluation.fixedSize else False,
        'dev': True if v == Evaluation.dev else False,
        'trainVsDev': True if v == Evaluation.trainVsDev else False,
        'trainVsTest': True if v == Evaluation.trainVsTest else False,
    })
    trues, evaluation = 0, ''
    for k in configuration['evaluation'].keys():
        if configuration['evaluation'][k] == True and type(True) == type(configuration['evaluation'][k]):
            evaluation = k.upper()
            trues += 1
    assert trues <= 1, 'There are more than one evaluation settings!'
    sys.stdout.write(tabs + 'Division: {0}'.format(evaluation) + doubleSep)


def setXPMode(v):
    configuration['xp'].update({
        'linear': True if v == XpMode.linear else False,
        'compo': True if v == XpMode.compo else False,
        'pytorch': True if v == XpMode.pytorch else False,
        'kiperwasser': True if v == XpMode.kiperwasser else False,
        'rnn': True if v == XpMode.rnn else False,
        'rnnNonCompo': True if v == XpMode.rnnNonCompo else False,
    })
    trues, mode = 0, ''
    for k in configuration['xp'].keys():
        if configuration['xp'][k] == True and type(True) == type(configuration['xp'][k]):
            mode = k.upper()
            trues += 1
    assert trues <= 1, 'There are more than one experimentation mode!'
    sys.stdout.write(tabs + 'Mode: {0}'.format(mode) + doubleSep)


def setDataSet(v):
    configuration['dataset'].update(
        {
            'sharedtask2': True if v == Dataset.sharedtask2 else False,
            'FTB': True if v == Dataset.FTB else False,
            'dimsum': True if v == Dataset.dimsum else False
        })
    trues, mode = 0, ''
    for k in configuration['dataset'].keys():
        if configuration['dataset'][k] and type(True) == type(configuration['dataset'][k]):
            ds = k.upper()
            trues += 1
    assert trues <= 1, 'Ambigious data set definition!'
    sys.stdout.write(tabs + 'Dataset: {0}'.format(ds) + doubleSep)
