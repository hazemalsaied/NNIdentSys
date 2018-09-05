#!/usr/bin/python
# -*- coding: utf-8 -*-

from asTrans import *


def parse(corpus):
    for sent in corpus:
        if len(sent.tokens) < 12:
            if not sent.isProjective():
                continue
            for t in sent.tokens:
                t.hasHead = False
            print printSent(sent)
            sent.initialTransition = Transition(None, sent, None, None, isInitial=True)
            transition = sent.initialTransition
            while not transition.isTerminal():
                print transition.type, '\t\t', transition.configuration
                transition = Next(transition)
            print transition.type, '\t\t', transition.configuration


def Next(trans):
    config = trans.configuration
    if isLeftArc(config):
        # config.stack[-1].dependencyParent = config.buffer[0].position
        # tokensWithAHead.add(config.stack[-1])
        newConfig = Configuration(
            list(config.stack[:-1]),
            config.buffer)
        newTrans = Transition(TransType.LEFT_ARC, trans.sent, trans, None, newConfig)
    elif isRightArc(config):
        newConfig = Configuration(
            list(config.stack[:-1]),
            list([config.stack[-1]] + config.buffer[1:]))
        newTrans = Transition(TransType.RIGHT_ARC, trans.sent, trans, None, newConfig)
    else:
        newStack = list(config.stack)
        newStack.append(config.buffer[0])
        newConfig = Configuration(
            newStack,
            list(config.buffer[1:]))
        newTrans = Transition(TransType.SHIFT, trans.sent, trans, None, newConfig)
    trans.next = newTrans
    return newTrans


def isRightArc(config, label=None):
    if config.stack and config.buffer and not config.buffer[0].hasHead:
        s0Pos = 0 if config.stack[-1] == 0 else config.stack[-1].position
        if (
                # (config.stack[-1] == 0 and 0 == config.buffer[0].dependencyParent and len(config.buffer) == 1) or
                (
                        # config.stack[-1] != 0 and
                        # config.stack[-1].position == config.buffer[0].dependencyParent
                        s0Pos == config.buffer[0].dependencyParent and (
                        not label or label == config.buffer[0].dependencyLabel)
                )):
            config.buffer[0].hasHead = True
            if label:
                return getRightArc(label)
            return True
    return False


def isLeftArc(config, label=None):
    if config.stack and config.buffer and config.stack[-1] != 0 and not config.stack[-1].hasHead and \
            config.stack[-1].dependencyParent == config.buffer[0].position and (
            not label or label == config.stack[-1].dependencyLabel):
        config.stack[-1].hasHead = True
        if label:
            return getRightArc(label)
        return True
    return False


def getRightArc(lbl):
    pass


def getLeftArc(lbl):
    pass


def printSent(sent):
    texts, parents = [], []
    for t in sent.tokens:
        texts.append(t.text)
        parents.append(t.dependencyParent)
    res = ''
    for i in range(len(texts)):
        res += '[{0}] {1} ({2}) '.format(i + 1, texts[i], parents[i])
    return res
