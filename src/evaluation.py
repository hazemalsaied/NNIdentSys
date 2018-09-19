from __future__ import division

import os
import sys

import reports
from config import configuration


def evaluate(corpus, categorization=True):
    tp, p, t, tpCat, pCat, tCat = getStatistics(corpus)
    scores = calculateScores(tp, p, t, 'Identification')
    if categorization:
        scores += calculateScores(tpCat, pCat, tCat, 'Categorization')
        catList = ['lvc', 'ireflv', 'vpc', 'id', 'oth']
        for cat in catList:
            tp, p, t = getCategoryStatistics(corpus, cat)
            scores += calculateScores(tp, p, t, cat)
    createMWEFiles(corpus)
    # reports.saveSettings()
    reports.saveScores(scores)
    corpus.analyzeTestSet()
    sys.stdout.write(reports.finalLine)
    return scores


def getStatistics(corpus):
    tp, p, t, tpCat, pCat, tCat = 0, 0, 0, 0, 0, 0

    for sent in corpus.testingSents:
        p += len(sent.vMWEs)
        t += len(sent.identifiedVMWEs)
        for vmw in sent.vMWEs:
            if len(vmw.tokens) > 1:
                pCat += 1
        for vmw in sent.identifiedVMWEs:
            if len(vmw.tokens) > 1:
                tCat += 1
        if sent.identifiedVMWEs:
            processedVmwe = []
            for m in sent.identifiedVMWEs:
                for vMWE in sent.vMWEs:
                    if m == vMWE and vMWE not in processedVmwe:
                        processedVmwe.append(vMWE)
                        tp += 1
                        if m.type == vMWE.type and len(vMWE.tokens) > 1:
                            tpCat += 1
    return tp, p, t, tpCat, pCat, tCat


def getCategoryStatistics(corpus, cat):
    tp, p, t = 0, 0, 0
    cat = cat.lower()
    for sent in corpus.testingSents:
        for vmw in sent.vMWEs:
            if vmw.type.lower() == cat:
                p += 1
        for vmw in sent.identifiedVMWEs:
            if vmw.type.lower() == cat:
                t += 1
        if not sent.identifiedVMWEs:
            continue
        processedVmwe = []
        for m in sent.identifiedVMWEs:
            for vMWE in sent.vMWEs:
                if m == vMWE and vMWE not in processedVmwe and m.type.lower() == vMWE.type.lower() \
                        and m.type.lower() == cat:
                    processedVmwe.append(vMWE)
                    tp += 1
    return tp, p, t


def getMWTStatistics(corpus):
    tp, p, t, tpCat, pCat, tCat = 0, 0, 0, 0, 0, 0

    for sent in corpus.testingSents:
        for vmw in sent.vMWEs:
            if len(vmw.tokens) == 1:
                p += 1
        for vmw in sent.identifiedVMWEs:
            if len(vmw.tokens) == 1:
                t += 1
        processedVmwe = []
        for m in sent.identifiedVMWEs:
            for vMWE in sent.vMWEs:
                if len(m.tokens) == 1 and m == vMWE and vMWE not in processedVmwe:
                    processedVmwe.append(vMWE)
                    tp += 1
                    if m.type.lower() == vMWE.type.lower():
                        tpCat += 1
    return tp, p, t, tpCat, p, t


def calculateScores(tp, p, t, title):
    '''

    :param tp: True positive
    :param p: positive
    :param t: true
    :param title: logging indicator
    :return: Fscore, recall, precision
    '''
    if p == 0 or t == 0 or tp == 0:
        if title == 'Identification':
            sys.stdout.write(reports.tabs + '{0} : {1}\n'.format(title, 0))
        return ['', 0, 0, 0]

    p = float(tp / p)
    r = float(tp / t)

    f = round(2 * (r * p) / (r + p), 3)
    r = round(r, 3)
    p = round(p, 3)
    sys.stdout.write(reports.tabs + '{0} : {1}\n'.format(title, f))
    return [title, f, r, p]


def createMWEFiles(corpus):
    # if not configuration['evaluation']['corpus']:
    #   return
    datasetConf, modelConf = configuration['dataset'], configuration['xp']
    dataset = 'ST2' if datasetConf['sharedtask2'] else \
        ('FTB' if datasetConf['sharedtask2'] else 'ST1')
    model = 'Linear' if modelConf['linear'] else (
        'Kiperwasser' if modelConf['kiperwasser'] else (
            'RNN' if modelConf['rnn'] else 'MLP'))

    folder = os.path.join(configuration['path']['projectPath'], configuration['path']['results'], dataset, model)
    if not os.path.exists(folder):
        os.makedirs(folder)
    predicted = corpus.toConllU()  # if dataset == 'ST2' else str(corpus)
    gold = corpus.toConllU(gold=True)  # if dataset == 'ST2' else corpus.getGoldenMWEFile()
    train = gold = corpus.toConllU(gold=True, train=True)  # if dataset == 'ST2' else corpus.getGoldenMWEFile()
    import datetime
    today = datetime.date.today().strftime("%Y.%B.%d.")
    with open(os.path.join(folder, today + corpus.langName + '.txt'), 'w') as f:
        f.write(predicted)
    with open(os.path.join(folder, today + corpus.langName + '.gold.txt'), 'w') as f:
        f.write(gold)
    with open(os.path.join(folder, today + corpus.langName + '.train.txt'), 'w') as f:
        f.write(train)
