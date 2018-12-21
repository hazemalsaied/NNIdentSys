from __future__ import division

import os
import sys

import reports
from config import configuration


def evaluate(sents, categorization=False, loggingg=True):
    tp, p, t, tpCat, pCat, tCat = getStatistics(sents)
    scores = calculateScores(tp, p, t, 'Identification', loggingg=loggingg)
    if categorization:
        scores += calculateScores(tpCat, pCat, tCat, 'Categorization', loggingg=loggingg)
        catList = ['lvc', 'ireflv', 'vpc', 'id', 'oth']
        for cat in catList:
            tp, p, t = getCategoryStatistics(sents, cat)
            scores += calculateScores(tp, p, t, cat, loggingg=loggingg)
    if loggingg:
        sys.stdout.write(reports.finalLine)
    # reports.saveSettings()
    # reports.saveScores(scores)
    # corpus.analyzeTestSet()

    return scores


def getStatistics(sents):
    tp, p, t, tpCat, pCat, tCat = 0, 0, 0, 0, 0, 0

    for sent in sents:
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


def getCategoryStatistics(sents, cat):
    tp, p, t = 0, 0, 0
    cat = cat.lower()
    for sent in sents:
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


def calculateScores(ig, g, i, title, loggingg=True):
    """

    :param logging:
    :param ig: golden identified
    :param g: golden
    :param i: identified
    :param title: logging indicator
    :return: Fscore, recall, precision
    """
    if g == 0 or i == 0 or ig == 0:
        if title == 'Identification' and loggingg:
            sys.stdout.write(reports.tabs + '{0} : {1}\n'.format(title, 0))
        return ['', 0, 0, 0]

    r = float(ig / g)
    p = float(ig / i)
    f = round(2 * (r * p) / (r + p), 3)
    r = round(r, 3)
    p = round(p, 3)
    if loggingg:
        sys.stdout.write(reports.tabs + '{0} : {1}\n'.format(title, f))
        sys.stdout.write(reports.tabs + 'P, R  : {0}, {1}\n'.format(p, r))
    return [title, f, r, p]

