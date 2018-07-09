import logging

import torch
from theano import function, config, shared, tensor

import modelCompo
import modelKiperwasser
import modelLinear
import modelLinearKeras as lkm
import modelNonCompo
import modelPytorch
import oracle
import reports
from corpus import *
from evaluation import evaluate
from normalisation import Normalizer
from parser import parse


def xp(langs=['FR'], train=False, corpus=False, cv=False, xpNum=5, title='', initSeed=0):
    verifyGPU()
    evlaConf = configuration["evaluation"]
    evlaConf["cluster"] = True
    global seed
    seed = initSeed
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if title:
        reports.createHeader(title)
    if not cv and not train:
        if corpus:
            ######################################
            #   Corpus
            ######################################
            evlaConf["debug"], evlaConf["corpus"] = False, True
            sys.stdout.write(reports.doubleSep + reports.tabs + 'Corpus Mode' + reports.doubleSep)
            for lang in langs:
                identify(lang)
        else:
            ######################################
            #   Debug
            ######################################
            evlaConf["debug"], evlaConf["train"] = True, False
            sys.stdout.write(reports.doubleSep + reports.tabs + 'Debug Mode' + reports.doubleSep)
            for lang in langs:
                identify(lang)
    elif train:
        ######################################
        #   Train
        ######################################
        evlaConf["debug"], evlaConf["train"] = False, train
        sys.stdout.write(reports.doubleSep + reports.tabs + 'Dev Mode' + reports.doubleSep)
        for lang in langs:
            for i in range(xpNum):
                numpy.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                identify(lang)
                seed += 1
    elif cv:
        ######################################
        #   CV Debug
        ######################################
        sys.stdout.write(reports.doubleSep + reports.tabs + 'Debug CV Mode' + reports.doubleSep)
        crossValidation(langs=langs, debug=True)
        ######################################
        #   CV
        ######################################
        sys.stdout.write(reports.doubleSep + reports.tabs + 'Debug Mode' + reports.doubleSep)
        for i in range(xpNum):
            seed += 1
            numpy.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            # crossValidation()
            ######################################
            #   Load
            ######################################
            # preTrainedPath= '/home/halsaied/nancy/NNIdenSys/NNIdenSys/Reports/FR-12/12-FR-modelWeigth.hdf5'
            # identify(load=configuration["evaluation"]["load"], loadFolderPath=loadFolderPath)


def identify(lang, loadFolderPath='', load=False):
    configuration["evaluation"]["load"] = load
    corpus = Corpus(lang)
    # corpus.getNewMWEPercentage()
    network, normalizer = parseAndTrain(corpus, loadFolderPath)
    if configuration["xp"]["linear"]:
        modelLinear.parse(corpus, network, normalizer)
    else:
        parse(corpus, network, normalizer)
    evaluate(corpus)


def parseAndTrain(corpus, loadFolderPath=''):
    if configuration["evaluation"]["load"]:
        # TODO rewrite the load part
        normalizer = reports.loadNormalizer(loadFolderPath)
        network = modelCompo.Network(normalizer)
        network.model = reports.loadModel(loadFolderPath)
    else:
        if not configuration["evaluation"]["cv"]["active"]:
            reports.createReportFolder(corpus.langName)
        oracle.parse(corpus)
        # corpus.getTransDistribution()
        if configuration["xp"]["linear"]:
            return modelLinear.train(corpus)
        normalizer = Normalizer(corpus)
        if configuration["xp"]["kiperwasser"]:
            network = modelKiperwasser.train(corpus, normalizer)
        elif configuration["xp"]["pytorch"]:
            network = modelPytorch.PytorchModel(normalizer)
            modelPytorch.main(network, corpus, normalizer)
        elif configuration["xp"]["compo"]:
            network = modelCompo.Network(normalizer)
            modelCompo.train(network.model, normalizer, corpus)
        else:
            network = modelNonCompo.Network(normalizer)
            modelNonCompo.train(network.model, normalizer, corpus)

    return network, normalizer


def crossValidation(langs=['FR'], debug=False):
    configuration["evaluation"]["cv"]["active"], scores, iterations = True, [0.] * 28, 5
    for lang in langs:
        reports.createReportFolder(lang)
        for cvIdx in range(configuration["evaluation"]["cv"]["currentIter"]):
            reports.createHeader('Iteration no.{0}'.format(cvIdx))
            configuration["evaluation"]["cv"]["currentIter"] = cvIdx
            cvCurrentIterFolder = os.path.join(reports.XP_CURRENT_DIR_PATH,
                                               str(configuration["evaluation"]["currentIter"]))
            if not os.path.isdir(cvCurrentIterFolder):
                os.makedirs(cvCurrentIterFolder)
            corpus = Corpus(lang)
            if debug:
                corpus.trainDataSet = corpus.trainDataSet[:100]
                corpus.testDataSet = corpus.testDataSet[:100]
            testRange, trainRange = corpus.getRangs()
            getTrainAndTestSents(corpus, testRange[cvIdx], trainRange[cvIdx])
            corpus.extractDictionaries()
            normalizer, network = parseAndTrain(corpus)
            parse(corpus, network, normalizer)
            tmpScores = evaluate(corpus)
            if len(tmpScores) != len(scores):
                print 'iter scores length are not equal!'
            for i in range(len(tmpScores)):
                if (isinstance(tmpScores[i], float) or isinstance(tmpScores[i], int)) and isinstance(scores[i], float):
                    scores[i] += float(tmpScores[i])
                elif tmpScores[i]:
                    scores[i] = tmpScores[i]
        reports.saveCVScores(scores)


def getTrainAndTestSents(corpus, testRange, trainRange):
    sent = corpus.trainDataSet
    corpus.testingSents = sent[testRange[0]:testRange[1]]
    if len(trainRange) == 2:
        corpus.trainingSents = sent[trainRange[0]:trainRange[1]]
    else:
        corpus.trainingSents = sent[trainRange[0]:trainRange[1]] + sent[trainRange[2]:trainRange[3]]


def identifyLinearKeras(langs=['FR'], ):
    sys.stdout.write(reports.doubleSep + reports.tabs + 'Linear model in KERAS' + reports.seperator)
    evlaConf = configuration["evaluation"]
    evlaConf["cluster"] = True
    evlaConf["debug"] = False
    evlaConf["debugTrainNum"] = 200
    evlaConf["train"] = True
    for lang in langs:
        corpus = Corpus(lang)
        oracle.parse(corpus)
        normalizer = lkm.Normalizer(corpus)
        network = lkm.LinearKerasModel(len(normalizer.tokens) + len(normalizer.pos))
        lkm.train(network.model, corpus, normalizer)
        parse(corpus, network, normalizer)
        evaluate(corpus)


def getAllLangStats(langs=['FR']):
    res = ''
    for lang in langs:
        corpus = Corpus(lang)
        res += corpus.langName + ',' + getStats(corpus.trainDataSet, asCSV=True) + ',' + \
               getStats(corpus.devDataSet, asCSV=True) + ',' + \
               getStats(corpus.testDataSet, asCSV=True) + '\n'
    return res


def analyzeCorporaAndOracle(langs):
    header = 'Non recognizable,Interleaving,Embedded,Distributed Embedded,Left Embedded,Right Embedded,Middle Embedded'
    analysisReport = header + '\n'
    for lang in langs:
        sys.stdout.write('Language = {0}\n'.format(lang))
        corpus = Corpus(lang)
        analysisReport += corpus.getVMWEReport() + '\n'
        oracle.parse(corpus)
        oracle.validate(corpus)
    with open('../Results/VMWE.Analysis.csv', 'w') as f:
        f.write(analysisReport)


def verifyGPU():
    vlen = 10 * 30 * 768
    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], tensor.exp(x))
    if numpy.any([isinstance(x.op, tensor.Elemwise) and
                  ('Gpu' not in type(x.op).__name__)
                  for x in f.maker.fgraph.toposort()]):
        sys.stdout.write(tabs + 'Attention: CPU used')
    else:
        sys.stdout.write(tabs + 'GPU Enabled')


# def identifyAttached(lang='FR'):
#     sys.stdout.write('*' * 20 + '\n')
#     sys.stdout.write('Deep model(No padding)\n')
#     corpus = Corpus(lang)
#     oracle.parse(corpus)
#     normalizer = Normalizer(corpus)
#     printReport(normalizer)
#     network = newNetwork.Network(normalizer)
#     newNetwork.train(network.model, normalizer, corpus)
#     parse(corpus, network, normalizer)
#     evaluate(corpus)
#     sys.stdout.write('*' * 20 + '\n')


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    # getScores('8-earlyStoppingErr', xpNum=10)
