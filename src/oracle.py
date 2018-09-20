#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import os

from corpus import getParents, isOrphanToken, getVMWEByTokens
from transitions import *

mwtDictionary = {}


def parse(corpus, printReport=False):
    printSentIdx, printSentNum = 0, 5
    report = ''
    global mwtDictionary
    mwtDictionary = corpus.mwtDictionary
    for sent in corpus:
        sent.initialTransition = Transition(isInitial=True, sent=sent)
        transition = sent.initialTransition
        while not transition.isTerminal():
            nextTransn = Next(transition.configuration)
            nextTransn.apply(transition, sent=transition.sent)
            transition = nextTransn
        if sent.containsEmbedding or sent.containsInterleaving:
            report += str(sent) + '\n'
        if printReport and len(sent.vMWEs) > 1 and printSentIdx < printSentNum:
            print sent
            printSentIdx += 1
    if report and False:
        directory = '../Results/Oracle/'
        fileName = '{0}-parsing.embed.interleaving.txt'.format(corpus.langName)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, fileName), 'w') as f:
            f.write(report)


def Next(config):
    newTransition = isMarkAs(config)
    if newTransition:
        return newTransition
    newTransition = isMerge(config)
    if newTransition:
        return newTransition
    newTransition = isReduce(config)
    if newTransition:
        return newTransition
    shift = Shift(sent=config.sent)
    return shift


def isMarkAs(config):
    s0Tokens = getTokens(config.stack[-1]) if config.stack else []
    if config.stack and not isIdentifiedVMWE(config.stack[-1]):
        # if s0Tokens and ((len(s0Tokens) == 1 and s0Tokens[-1] in mwtDictionary) or not isIdentifiedVMWE(config.stack[-1])):
        # if s0Tokens and len(s0Tokens) == 1 and s0Tokens[-1].getLemma() in mwtDictionary:
        #    pass
        # s0Tokens = getTokens(config.stack[-1])
        selectedParents = getParents(s0Tokens)
        if selectedParents:
            if len(selectedParents) == 1:
                selectedParents[0].parsedByOracle = True
                newTrans = MarkAs(type=getMWTTypeFromStr(selectedParents[0].type), sent=config.sent)
                return newTrans
            else:
                if selectedParents[0].tokens[-1] == s0Tokens[-1]:
                    newTrans = MarkAs(type=getMWTTypeFromStr(selectedParents[0].type), sent=config.sent)
                    return newTrans
    return None


def isMerge(config):
    if len(config.stack) > 1:
        tokens = getTokens(config.stack[-1]) + getTokens(config.stack[-2])
        sharedParents = getParents(tokens, allChildren=False)
        if len(sharedParents) > 1:
            if sharedParents[0].isLeftEmbedded and not sharedParents[0].parsedByOracle:
                return Merge(sent=config.sent)
            if sharedParents[0].isRightEmbedded and not sharedParents[0].parsedByOracle:
                return Merge(sent=config.sent)
            if sharedParents[0].isMiddleEmbedded and not sharedParents[0].parsedByOracle:
                # sharedParents[0].getTokenPositionString() in VMWE(0,tokens=tokens).getTokenPositionString():
                return Merge(sent=config.sent)
            if sharedParents[0].parsedByOracle:
                return Merge(sent=config.sent)
            if sharedParents[0].isRightEmbedded and getParents(tokens) and getParents(tokens)[0].isRightEmbedded:
                return Merge(sent=config.sent)
        if len(sharedParents) == 1 and (not sharedParents[0].isRightEmbedder or (
                    sharedParents[0].isRightEmbedder and sharedParents[0].child.parsedByOracle)) and (
                    not (tokens[0].parentMWEs[0].isMiddleEmbedded and tokens[0].parentMWEs[0].isRecognizable) or (
                                tokens[0].parentMWEs[0].isMiddleEmbedded and tokens[0].parentMWEs[0].isRecognizable and
                            tokens[0].parentMWEs[0].parsedByOracle)):
            return Merge(sent=config.sent)
    return None


def isReduce(config):
    reduce = Reduce(sent=config.sent)
    # Orphan Token on Stack
    if config.stack:
        if str(config.stack[-1].__class__) == 'corpus.Token' and isOrphanToken(config.stack[-1]):
            return reduce
        # Identified VMWE on Stack
        vmwe = getVMWEByTokens(getTokens(config.stack[-1]))
        if isIdentifiedVMWE(config.stack[-1]) and (
                    not vmwe.isEmbedded or vmwe.isEmbedded and vmwe.parent.parsedByOracle):
            return reduce
    # Empy Buffer With Full Stack
    if not config.buffer and config.stack:
        return reduce
    return None


oracleReportPath = '../Results/Oracle'


def validate(corpus):
    report = ''
    sentNum = 0
    for sent in corpus.trainingSents:
        MergeTransNum = 0
        markAsSet = {TransitionType.MARK_AS_ID, TransitionType.MARK_AS_OTH, TransitionType.MARK_AS_VPC,
                     TransitionType.MARK_AS_LVC,
                     TransitionType.MARK_AS_IREFLV}
        trans = sent.initialTransition
        while trans:
            if trans.type in markAsSet:
                MergeTransNum += 1
            trans = trans.next
        if len(sent.getRecognizableVMWEs()) != MergeTransNum:
            sentNum += 1
            report += str(sent) + '\n'
    if report:
        logging.error('ATTENTION: Oracle problems with {0} sentences!'.format(sentNum))
        with open(os.path.join(oracleReportPath, corpus.langName + '.txt'), 'w') as oracleReport:
            oracleReport.write(report)


def isIdentifiedVMWE(element):
    return isinstance(element, list) and len(element) == 1
