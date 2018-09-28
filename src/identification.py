import datetime
import logging

# import torch
from theano import function, config, shared, tensor

import modelCompo
import modelNonCompo
# import modelPytorch
import modelRnn
import modelRnnNonCompo
import oracle
import reports
from corpus import *
from evaluation import evaluate
from normalisation import Normalizer
from parser import parse


def identify(lang):
    corpus = Corpus(lang)
    oracle.parse(corpus)
    startTime = datetime.datetime.now()
    network, normalizer = parseAndTrain(corpus)
    sys.stdout.write(reports.doubleSep + reports.tabs + 'Training time : {0}'.
                     format(datetime.datetime.now() - startTime) + reports.doubleSep)
    parse(corpus, network, normalizer)
    reports.printParsedSents(corpus, 1)
    evaluate(corpus)

def parseAndTrain(corpus):
    if configuration['xp']['rnn']:
        network = modelRnn.Network(corpus)
        modelRnn.train(network, corpus)
        return network, None
    if configuration['xp']['rnnNonCompo']:
        network = modelRnnNonCompo.Network(corpus)
        modelRnnNonCompo.train(network, corpus)
        return network, None
    # if configuration['xp']['kiperwasser']:
    #     network = modelKiperwasser.train(corpus)
    #     normalizer = Normalizer(corpus)
    #     return network, normalizer
    normalizer = Normalizer(corpus)
    # if configuration['xp']['pytorch']:
    #     network = modelPytorch.PytorchModel(normalizer)
    #     modelPytorch.main(network, corpus, normalizer)
    if configuration['xp']['compo']:
        network = modelCompo.Network(normalizer)
        modelCompo.train(network.model, normalizer, corpus)
    else:
        network = modelNonCompo.Network(normalizer)
        modelNonCompo.train(network.model, normalizer, corpus)
    return network, normalizer


def crossValidation(langs, debug=False):
    configuration['evaluation']['cv']['active'], scores, iterations = True, [0.] * 28, 5
    for lang in langs:
        reports.createReportFolder(lang)
        for cvIdx in range(configuration['evaluation']['cv']['currentIter']):
            reports.createHeader('Iteration no.{0}'.format(cvIdx))
            configuration['evaluation']['cv']['currentIter'] = cvIdx
            cvCurrentIterFolder = os.path.join(reports.XP_CURRENT_DIR_PATH,
                                               str(configuration['evaluation']['currentIter']))
            if not os.path.isdir(cvCurrentIterFolder):
                os.makedirs(cvCurrentIterFolder)
            corpus = Corpus(lang)
            if debug:
                corpus.trainDataSet = corpus.trainDataSet[:100]
                corpus.testDataSet = corpus.testDataSet[:100]
            testRange, trainRange = corpus.getRangs()
            getTrainAndTestSents(corpus, testRange[cvIdx], trainRange[cvIdx])
            corpus.extractDictionaries()
            oracle.parse(corpus)
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
    corpus.trainingSents = sent[trainRange[0]:trainRange[1]] if len(trainRange) == 2 else \
        sent[trainRange[0]:trainRange[1]] + sent[trainRange[2]:trainRange[3]]


def getAllLangStats(langs):
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







if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
