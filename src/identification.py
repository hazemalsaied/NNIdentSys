import datetime
import logging

import modelCompactKiper
import modelKiperwasser
import modelLinear
import modelNonCompo
import modelRnn
import modelRnnNonCompo
import oracle
import reports
from corpus import *
from evaluation import evaluate
from parser import parse


# from analysis import errorAnalysis, exportEnalysis

def identify(lang):
    corpus = Corpus(lang)
    oracle.parse(corpus)
    startTime = datetime.datetime.now()
    network, vectorizer = parseAndTrain(corpus)
    sys.stdout.write('{0}{1}Training time : {2}{3}'.
                     format(reports.doubleSep, reports.tabs,
                            datetime.datetime.now() - startTime, reports.doubleSep))
    startTime = datetime.datetime.now()
    parse(corpus.testingSents, network, vectorizer)
    sys.stdout.write('{0}{1}Parsing time : {2}{3}'.
                     format(reports.doubleSep, reports.tabs,
                            datetime.datetime.now() - startTime, reports.doubleSep))
    evaluate(corpus.testingSents)
    corpus.createMWEFiles()
    if configuration['others']['printTest']:
        for s in corpus.testingSents:
            s.initialTransition = None
            print s
    sys.stdout.flush()
    return corpus


def identifyWithBoth(lang):
    from xpTools import setXPMode, XpMode
    setXPMode(XpMode.linear)
    corpus = Corpus(lang)
    oracle.parse(corpus)
    startTime = datetime.datetime.now()
    configuration['features'].update({
        'lemma': True,
        'token': True,
        'pos': True,
        'suffix': False,
        'b1': True,
        'bigram': True,
        's0b2': True,
        'trigram': True,
        'syntax': False,
        'syntaxAbstract': False,
        'dictionary': True,
        's0TokenIsMWEToken': True,
        's0TokensAreMWE': False,
        'history1': True,
        'history2': False,
        'history3': True,
        'stackLength': False,
        'distanceS0s1': True,
        'distanceS0b0': True
    })
    linearModel, linearVectorizer = parseAndTrain(corpus)
    sys.stdout.write('{0}{1}Training time : {2}{3}'.
                     format(reports.doubleSep, reports.tabs,
                            datetime.datetime.now() - startTime, reports.doubleSep))
    setXPMode(None)
    configuration['sampling'].update({
        'importantSentences': True,
        'overSampling': True,
        'sampleWeight': True,
        'favorisationCoeff': 6,
        'focused': True})
    startTime = datetime.datetime.now()
    mlpModel, mlpVectorizer = parseAndTrain(corpus)
    sys.stdout.write('{0}{1}Training time : {2}{3}'.
                     format(reports.doubleSep, reports.tabs,
                            datetime.datetime.now() - startTime, reports.doubleSep))

    startTime = datetime.datetime.now()
    setXPMode(XpMode.linear)
    parse(corpus.testingSents, linearModel, linearVectorizer)
    setXPMode(None)
    parse(corpus.testingSents, mlpModel, mlpVectorizer, initialize=False)
    sys.stdout.write('{0}{1}Parsing time : {2}{3}'.
                     format(reports.doubleSep, reports.tabs,
                            datetime.datetime.now() - startTime, reports.doubleSep))
    if configuration['others']['complInter']:
        getIntersectedMWEs(corpus)
    elif configuration['others']['complFreq']:
        getMWEsAccFrequency(corpus)
    evaluate(corpus.testingSents)

    # corpus.createMWEFiles()
    # if configuration['others']['printTest']:
    #     for s in corpus.testingSents:
    #         s.initialTransition = None
    #         print s
    sys.stdout.flush()
    return corpus


def getIntersectedMWEs(corpus):
    for s in corpus.testingSents:
        nonCorrectMWEIdxs = []
        for v in s.identifiedVMWEs:
            if v.predictingModel == 'linear' or v.predictingModel == 'mlp':
                nonCorrectMWEIdxs.append(s.identifiedVMWEs.index(v))
        for idx in sorted(nonCorrectMWEIdxs, reverse=True):
            s.identifiedVMWEs.pop(idx)


def getMWEsAccFrequency(corpus):
    for s in corpus.testingSents:
        nonCorrectMWEIdxs = []
        for v in s.identifiedVMWEs:
            partiallySeen = v.getLemmaString() not in corpus.mweDictionary
            barelySeen = v.getLemmaString() in corpus.mweDictionary and corpus.mweDictionary[v.getLemmaString()] <= 5
            if v.predictingModel in ['linear', 'mlp'] and (partiallySeen or barelySeen):
                nonCorrectMWEIdxs.append(s.identifiedVMWEs.index(v))
        for idx in sorted(nonCorrectMWEIdxs, reverse=True):
            s.identifiedVMWEs.pop(idx)


def identifyWithLinearInMlp(lang):
    configuration['others']['verbose'] = False
    linearModels, linearVecs = jackknifing(lang, True)
    configuration['others']['verbose'] = True
    configuration['xp']['linear'] = False
    configuration['sampling'].update({
        'overSampling': True,
        'importantSentences': True,
        'sampleWeight': True,
        'favorisationCoeff': 6,
        'focused': True
    })
    corpus = Corpus(lang)
    oracle.parse(corpus)
    startTime = datetime.datetime.now()
    model = modelNonCompo.Network(corpus, linearInMLP=True)
    model.train(corpus, linearModels=linearModels, linearNormalizers=linearVecs)
    sys.stdout.write('{0}MLP training time : {1}{2}'.
                     format(reports.tabs, datetime.datetime.now() - startTime, reports.doubleSep))
    startTime = datetime.datetime.now()
    parse(corpus.testingSents, model, linearModels=linearModels, linearVecs=linearVecs)
    sys.stdout.write('{0}{1}Parsing time : {2}{3}'.
                     format(reports.doubleSep, reports.tabs, datetime.datetime.now() - startTime, reports.doubleSep))
    evaluate(corpus.testingSents)
    configuration['xp']['linear'] = False
    sys.stdout.flush()


def identifyWithMlpInLinear(lang):
    configuration['others']['verbose'] = False
    mlpModels, mlpNormalizers = jackknifing(lang, False)
    configuration['others']['verbose'] = True
    configuration['sampling'].update({
        'overSampling': False,
        'importantSentences': False,
        'importantTransitions': False,
        'sampleWeight': False,
        'favorisationCoeff': 1,
        'focused': False
    })
    corpus = Corpus(lang)
    oracle.parse(corpus)
    startTime = datetime.datetime.now()
    sys.stdout.flush()
    linearModel, linearVec = modelLinear.train(corpus, mlpModels=mlpModels)
    sys.stdout.flush()
    sys.stdout.write('{0}Linear training time : {1}{2}'.
                     format(reports.tabs, datetime.datetime.now() - startTime, reports.doubleSep))
    configuration['xp']['linear'] = True
    startTime = datetime.datetime.now()
    sys.stdout.flush()
    parse(corpus.testingSents, linearModel, linearVec, mlpModels=mlpModels)
    sys.stdout.write('{0}{1}Parsing time : {2}{3}'.
                     format(reports.doubleSep, reports.tabs, datetime.datetime.now() - startTime, reports.doubleSep))
    evaluate(corpus.testingSents)
    configuration['xp']['linear'] = False
    sys.stdout.flush()


def jackknifing(lang, linear=True):
    if linear:
        configuration['xp']['linear'] = True
        configuration['sampling'].update({
            'overSampling': False,
            'importantSentences': False,
            'importantTransitions': False,
            'sampleWeight': False,
            'favorisationCoeff': 1,
            'focused': False
        })
    else:
        configuration['xp']['linear'] = False
        configuration['sampling'].update({
            'overSampling': True,
            'importantSentences': False,
            'importantTransitions': False,
            'sampleWeight': True,
            'favorisationCoeff': 6,
            'focused': True
        })
    models, normalizers = dict(), dict()
    for i in range(5):
        models[i], normalizers[i] = jackknifingAFold(lang, i, linear=linear)
        sys.stdout.write('Finished training the fold {0}\n'.format(i))
    models[5], normalizers[5] = jackknifingAFold(lang, linear=linear, all=True)
    sys.stdout.write('Finished training the whole corpus \n')
    return models, normalizers


def jackknifingAFold(lang, foldIdx=-1, linear=True, all=False):
    corpus = Corpus(lang)
    if not all:
        foldLength = int(len(corpus.trainingSents) / 5)
        foldStart = foldIdx * foldLength
        foldEnd = foldStart + foldLength
        corpus.testingSents = corpus.trainingSents[foldStart:foldEnd]
        corpus.trainingSents = corpus.trainingSents[:foldStart] + corpus.trainingSents[foldEnd:]
    if not linear:
        configuration['sampling']['importantSentences'] = True
        corpus.trainingSents = corpus.filterImportatntSents()
    oracle.parse(corpus)
    vectorizer = None
    if linear:
        network, vectorizer = modelLinear.train(corpus)
    else:
        network = modelNonCompo.Network(corpus)
        network.train(corpus)
    # parse(corpus.testingSents, network, vectorizer)
    # evaluate(corpus.testingSents)
    # corpus.testingSents = corpus.trainingSents
    # parse(corpus.testingSents, network, vectorizer)
    # evaluate(corpus.testingSents)

    return network, vectorizer


def parseAndTrain(corpus):
    if configuration['xp']['linear']:
        return modelLinear.train(corpus)
    if configuration['xp']['rnn']:
        network = modelRnn.Network(corpus)
        modelRnn.train(network, corpus)
        return network, None
    if configuration['xp']['rnnNonCompo']:
        network = modelRnnNonCompo.Network(corpus)
        modelRnnNonCompo.train(network, corpus)
        return network, None
    if configuration['xp']['kiperwasser']:
        network = modelKiperwasser.train(corpus, configuration)
        return network, None
    if configuration['xp']['kiperComp']:
        network = modelCompactKiper.train(corpus, configuration)
        return network, None
    network = modelNonCompo.Network(corpus)
    network.train(corpus)
    return network, None


def crossValidation(langs, debug=False):
    configuration['evaluation']['cv'], scores, iterations = True, [0.] * 28, 5
    for lang in langs:
        for cvIdx in range(configuration['others']['cvFolds']):
            reports.createHeader('Iteration no.{0}'.format(cvIdx))
            configuration['others']['currentIter'] = cvIdx
            cvCurrentIterFolder = os.path.join(reports.XP_CURRENT_DIR_PATH,
                                               str(configuration['others']['currentIter']))
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
        # reports.saveCVScores(scores)


def getTrainAndTestSents(corpus, testRange, trainRange):
    sent = corpus.trainDataSet
    corpus.testingSents = sent[testRange[0]:testRange[1]]
    corpus.trainingSents = sent[trainRange[0]:trainRange[1]] if len(trainRange) == 2 else \
        sent[trainRange[0]:trainRange[1]] + sent[trainRange[2]:trainRange[3]]


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
