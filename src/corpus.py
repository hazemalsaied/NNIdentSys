#!/usr/bin/python
# -*- coding: utf-8 -*-
import itertools
import logging
import os
import pickle
import random
import sys

import reports
# from config from config import configuration
from config import configuration


class Corpus:
    """
        a class used to encapsulate all the information of the corpus
    """

    def __init__(self, langName):
        """
            an initializer of the corpus, responsible of creating a structure of objects encapsulating all the
            information of the corpus, its sentences, tokens and MWEs.

            This function iterate over the lines of corpus document to create the precedent ontology
        """
        sys.stdout.write('# Language = {0}\n'.format(langName))
        self.langName = langName
        self.trainingSents, self.testingSents = [], []
        self.mweDictionary, self.mwtDictionary, self.mweTokenDictionary = dict(), dict(), dict()
        path = os.path.join(configuration["path"]["projectPath"], configuration["path"]["corpusRelativePath"], langName)
        # print 'corpora path', path
        mweFile, testMweFile = os.path.join(path, 'train.parsemetsv'), os.path.join(path, 'test.parsemetsv')
        conlluFile, testConllu = getTrainAndTestConlluPath(path)
        if conlluFile and testConllu:
            # if not configuration["evaluation"]["load"]:
            self.trainDataSet = readConlluFile(conlluFile)
            integrateMweFile(mweFile, self.trainDataSet)
            self.testDataSet = readConlluFile(testConllu)
            integrateMweFile(testMweFile, self.testDataSet)
        else:
            # if not configuration["evaluation"]["load"]:
            self.trainDataSet = readMWEFile(mweFile)
            self.testDataSet = readMWEFile(testMweFile)
        # if not configuration["evaluation"]["load"]:
        self.analyzeSents()
        # Sorting parents to get the direct parent on the top of parentVMWEs list
        self.text = self.toText()
        self.orderParentVMWEs()
        if not configuration["evaluation"]["cv"]["active"]:
            self.shuffleSent(langName)
            self.distributeSent(langName)
            self.getTrainAndTest()
            self.extractDictionaries()

    def extractDictionaries(self):
        self.mweDictionary = self.getMWEDictionary()
        self.mwtDictionary = self.getMWEDictionary(mwtOnly=True)
        self.mweTokenDictionary = self.getMWETokenDictionary()

    def toText(self):
        res = ''
        for sent in self:
            res += sent.text + '\n'
        return res

    def getMWEDictionary(self, mwtOnly=False):
        mweDictionary = {}
        for sent in self:
            for mwe in sent.vMWEs:
                if mwtOnly:
                    if len(mwe.tokens) != 1:
                        continue
                lemmaString = mwe.getLemmaString()
                if lemmaString in mweDictionary:
                    mweDictionary[lemmaString] += 1
                else:
                    mweDictionary[lemmaString] = 1
        return mweDictionary

    def getMWETokenDictionary(self):
        mweTokenDictionary = {}
        for sent in self:
            for mwe in sent.vMWEs:
                for token in mwe.tokens:
                    mweTokenDictionary[token.text.lower()] = True
        return mweTokenDictionary

    def divideCorpus(self, foldIdx, foldNum=5):
        testFoldSize = int(len(self.trainDataSet) * 0.2)
        self.testingSents = self.trainDataSet[foldIdx * testFoldSize: (foldIdx + 1) * testFoldSize]
        if foldIdx == 0:
            self.trainingSents = self.trainDataSet[(foldIdx + 1) * testFoldSize:]
        elif foldIdx == foldNum - 1:
            self.trainingSents = self.trainDataSet[:foldIdx * testFoldSize]
        else:
            self.trainingSents = itertools.chain(self.trainDataSet[0:foldIdx * testFoldSize],
                                                 self.trainDataSet[(foldIdx + 1) * testFoldSize:])

    def getGoldenMWEFile(self):
        res = ''
        for sent in self.testingSents:
            idx = 1
            for token in sent.tokens:
                tokenLbl = ''
                if token.parentMWEs:
                    for parent in token.parentMWEs:
                        if tokenLbl:
                            tokenLbl += ';' + str(parent.id)
                        else:
                            tokenLbl += str(parent.id)
                        if token.getLemma() == parent.tokens[0].getLemma() and parent.type:
                            tokenLbl += ':' + parent.type
                if tokenLbl == '':
                    tokenLbl = '_'
                res += '{0}\t{1}\t{2}\t{3}\n'.format(idx, token.text.strip(), '_', tokenLbl)
                idx += 1
            res += '\n'
        return res

    def getMWEDictionaryWithWindows(self):
        mweDictionary = {}
        for sent in self:
            for mwe in sent.vMWEs:
                windows = ''
                for i in range(len(mwe.tokens)):
                    if i > 0:
                        distance = str(sent.tokens.index(mwe.tokens[i]) - sent.tokens.index(mwe.tokens[i - 1]))
                        if windows:
                            windows += ';' + distance
                        else:
                            windows = distance
                lemmaString = mwe.getLemmaString()
                if lemmaString in mweDictionary and mweDictionary[lemmaString] != windows:
                    oldWindow = mweDictionary[lemmaString]
                    oldWindowDistances = oldWindow.split(';')
                    newWindowDistances = windows.split(';')
                    newWindows = ''
                    if len(oldWindowDistances) == len(newWindowDistances):
                        for i in range(len(oldWindowDistances)):
                            if oldWindowDistances[i] > newWindowDistances[i]:
                                newWindows += oldWindowDistances[i] + (';' if i < (len(oldWindowDistances) - 1) else '')
                            else:
                                newWindows += newWindowDistances[i] + (';' if i < (len(newWindowDistances) - 1) else '')
                    mweDictionary[lemmaString] = newWindows
                else:
                    mweDictionary[lemmaString] = windows
        return mweDictionary

    def getTrainAndTest(self):
        evalConfig = configuration["evaluation"]
        if evalConfig["debug"]:
            debugTrainNum = evalConfig["debugTrainNum"]
            self.trainingSents =self.trainDataSet[:debugTrainNum]  #getVMWESents(self.trainDataSet, debugTrainNum)
            self.testingSents = self.trainDataSet[debugTrainNum:debugTrainNum*2]#getVMWESents(self.testDataSet, debugTrainNum)
        elif evalConfig["train"]:
            pointer = int(len(self.trainDataSet) * (1 - evalConfig["test"]))
            self.trainingSents = self.trainDataSet[:pointer]
            self.testingSents = self.trainDataSet[pointer:]
        elif evalConfig["corpus"]:
            self.trainingSents = self.trainDataSet
            self.testingSents = self.testDataSet

        sys.stdout.write('# Train = {0}\n'.format(len(self.trainingSents)))
        sys.stdout.write('# Test = {0}\n'.format(len(self.testingSents)))

    def distributeSent(self,lang):
        if configuration["evaluation"]["shuffleTrain"]:
            return
        print lang +'idxDic'
        idxDic = loadObj(lang +'idxDic')
        newTrainingSentSet = []
        for key in idxDic.keys():
            newTrainingSentSet.append(self.trainDataSet[idxDic[key]])
        self.trainDataSet = newTrainingSentSet

    def shuffleSent(self, lang):
        if not configuration["evaluation"]["shuffleTrain"]:
            return
        idxDic = dict()
        idxList = range(len(self.trainDataSet))
        random.shuffle(idxList)
        for i in range(len(self.trainDataSet)):
            idxDic[i] = idxList[i]
        newTrainingSentSet = []
        for key in idxDic.keys():
            newTrainingSentSet.append(self.trainDataSet[idxDic[key]])
        self.trainDataSet = newTrainingSentSet
        saveObj(idxDic, lang+'idxDic')

    def analyzeSents(self):
        for sent in self.trainDataSet:
            sent.recognizeEmbedded()
            sent.recognizeInterleaving()
            sent.recognizeAlternating()
            sent.setTokenParent()

    def orderParentVMWEs(self):
        for sent in self.trainDataSet:
            if sent.vMWEs and sent.containsEmbedding:
                for token in sent.tokens:
                    if token.parentMWEs and len(token.parentMWEs) > 1:
                        token.parentMWEs = sorted(token.parentMWEs, key=lambda mwe: (len(mwe)))

    def getVMWEReport(self):
        embeddedNum, leftEmbeddedNum, rightEmbeddedNum, middleEmbeddedNum = 0, 0, 0, 0
        nonRecognizableNum, distributedEmbeddedNum, interleavingNum = 0, 0, 0
        # interleavingReport, embeddingReport = '', ''
        for sent in self.trainDataSet:
            if len(sent.vMWEs) > 1:
                for vmwe in sent.vMWEs:
                    nonRecognizableNum += 1 if not vmwe.isRecognizable else 0
                    # interleavingReport += str(sent) if vmwe.isInterleaving else ''
                    interleavingNum += 1 if vmwe.isInterleaving else 0
                    # embeddingReport += str(sent) if vmwe.isEmbedded else ''
                    embeddedNum += 1 if vmwe.isEmbedded else 0
                    distributedEmbeddedNum += 1 if vmwe.isEmbedded and not vmwe.isRecognizable else 0
                    leftEmbeddedNum += 1 if vmwe.isLeftEmbedded else 0
                    rightEmbeddedNum += 1 if vmwe.isRightEmbedded else 0
                    middleEmbeddedNum += 1 if vmwe.isMiddleEmbedded else 0
        res = '{0},{1},{2},{3},{4},{5},{6}'.format(nonRecognizableNum, interleavingNum,
                                                   embeddedNum, distributedEmbeddedNum, leftEmbeddedNum,
                                                   rightEmbeddedNum, middleEmbeddedNum)
        print res
        return res

    def getRangs(self):
        sents = self.trainDataSet
        testNum = int(len(sents) * 0.2)
        testRanges = [[0, testNum], [testNum, 2 * testNum], [2 * testNum, 3 * testNum], [3 * testNum, 4 * testNum],
                      [4 * testNum, len(sents)]]

        trainRanges = [[testNum, len(sents)], [0, testNum, 2 * testNum, len(sents)],
                       [0, 2 * testNum, 3 * testNum, len(sents)], [0, 3 * testNum, 4 * testNum, len(sents)],
                       [0, 4 * testNum]]

        return testRanges, trainRanges

    def __str__(self):
        res = ''
        for sent in self.testingSents:
            tokenList = []
            for token in sent.tokens:
                tokenList.append(token.text.strip())
            labels = ['_'] * len(tokenList)
            for mwe in sent.identifiedVMWEs:
                for token in mwe.tokens:
                    if labels[token.position - 1] == '_':
                        labels[token.position - 1] = str(mwe.id)
                    else:
                        labels[token.position - 1] += ';' + str(mwe.id)
                    if mwe.tokens[0] == token and mwe.type:
                        labels[token.position - 1] += ':' + mwe.type

            for i in range(len(tokenList)):
                res += '{0}\t{1}\t{2}\t{3}\n'.format(i + 1, tokenList[i], '_', labels[i])
            res += '\n'
        return res

    def __iter__(self):
        for sent in self.trainingSents:
            yield sent


class Sentence:
    """
       a class used to encapsulate all the information of a sentence
    """

    def __init__(self, idx, sentid=''):

        self.sentid = sentid
        self.id = idx
        self.text = ''
        self.tokens = []
        self.vMWEs = []
        self.identifiedVMWEs = []
        self.initialTransition = None
        self.containsEmbedding = False
        self.containsInterleaving = False
        self.containsDistributedEmbedding = False

    def getWMWEIds(self):
        result = []
        for vMWE in self.vMWEs:
            result.append(vMWE.getId())
        return result

    def getVMWE(self, idx):

        for vMWE in self.vMWEs:
            if vMWE.getId() == int(idx):
                return vMWE
        return None

    def setTextandPOS(self):

        tokensTextList = []
        for token in self.tokens:
            self.text += token.text + ' '
            tokensTextList.append(token.text)
        self.text = self.text.strip()

    def recognizeEmbedded(self):
        for vMwe1 in self.vMWEs:
            if vMwe1.isEmbedded:
                continue
            for vMwe2 in self.vMWEs:
                if vMwe1 is not vMwe2 and vMwe1.getTokenPositionString() == vMwe2.getTokenPositionString():
                    vMwe1.isRecognizable = False
                    vMwe1.isBadAnnotated = True
                if vMwe1 is not vMwe2 and len(vMwe1) < len(vMwe2):
                    isEmbedded = True
                    vMwe2Lemma = vMwe2.getTokenPositionString()
                    for token in vMwe1.tokens:
                        if token.getPositionString() not in vMwe2Lemma:
                            isEmbedded = False
                            break
                    if isEmbedded:
                        self.containsEmbedding = True
                        vMwe1.isEmbedded = True
                        vMwe2.isEmbedder = True
                        vMwe1.parent = vMwe2
                        vMwe2.child = vMwe1
                        if vMwe2.getTokenPositionString().startswith(vMwe1.getTokenPositionString()):
                            vMwe1.isLeftEmbedded = True
                            vMwe2.isLeftEmbedder = True
                        elif vMwe2.getTokenPositionString().endswith(vMwe1.getTokenPositionString()):
                            vMwe1.isRightEmbedded = True
                            vMwe2.isRightEmbedder = True
                        elif vMwe1.getTokenPositionString() in vMwe2.getTokenPositionString():
                            vMwe1.isMiddleEmbedded = True
                            vMwe2.isMiddleEmbedder = True
                        else:
                            vMwe1.isRecognizable = False

    def recognizeInterleaving(self):
        # @TODO: mainpulatr triple interleaving: Example CS
        # '25356- :: prakticky všechno 2:3:1 **se** 2 **zdá** na burze 3:1 **obracet** 1 **k** 1 **lepšímu** .'
        # 'MWEs:'
        # '1- ID: se obracet k dobrý (+):Embedder'
        # '2- IReflV: se zdát (-):Interleaving'
        # '3- IReflV: se obracet (+):Left Embedded'
        processedVMWEs = []
        for vmwe1 in self.vMWEs:
            if vmwe1 not in processedVMWEs:
                for vmwe2 in self.vMWEs:
                    if vmwe1 is not vmwe2:
                        firstCondition, secondCondition = False, False
                        master = vmwe1 if len(vmwe1) > len(vmwe2) else vmwe2
                        slave = vmwe1 if len(vmwe1) <= len(vmwe2) else vmwe2
                        for token in slave.tokens:
                            if master in token.parentMWEs:
                                firstCondition = True
                            else:
                                secondCondition = True
                        if firstCondition and secondCondition:
                            self.containsInterleaving = True
                            processedVMWEs.extend([slave])  # master,
                            master.isInterleaver = True
                            slave.isInterleaving = True
                            slave.isRecognizable = False

    def recognizeAlternating(self):
        processedVMWEs = []
        for vmwe1 in self.vMWEs:
            if vmwe1 not in processedVMWEs:
                for vmwe2 in self.vMWEs:
                    if vmwe1 is vmwe2:
                        continue
                    firstCondition = True
                    for token in vmwe1.tokens:
                        if token.getPositionString() in vmwe2.getTokenPositionString():
                            firstCondition = False
                    if firstCondition:
                        start = vmwe1.tokens[0].position
                        end = vmwe1.tokens[-1].position
                        secondCondition = False
                        for token in vmwe2.tokens:
                            if end > token.position > start:
                                secondCondition = True
                        if secondCondition:
                            orderedList = []
                            for token in vmwe1.tokens:
                                positionStr = '0' + str(token.position) if token.position < 10 else str(token.position)
                                orderedList.append(positionStr + '-1')
                            for token in vmwe2.tokens:
                                positionStr = '0' + str(token.position) if token.position < 10 else str(token.position)
                                orderedList.append(positionStr + '-2')
                            orderedList = sorted(orderedList)
                            condA, condB = False, False
                            for i in range(len(orderedList)):
                                if orderedList[i].endswith('-2') and 0 < i < len(orderedList) - 1:
                                    if orderedList[i - 1].endswith('-1') and orderedList[i + 1].endswith('-1'):
                                        condB = True
                            if condB:
                                vmwe2.isRecognizable = False
                                vmwe2.isAletrnating = True
                                processedVMWEs.extend([vmwe1, vmwe2])

    def getDirectParents(self):
        for token in self.tokens:
            token.getDirectParent()

    def getInterleavingVMWEs(self):
        res = []
        for mwe in self.vMWEs:
            if mwe.isInterleaving:
                res.append(mwe)
        return res

    def getEmbeddedVMWEs(self):
        res = []
        for mwe in self.vMWEs:
            if mwe.isEmbedded:
                res.append(mwe)
        return res

    def getLeftEmbeddedVMWEs(self):
        res = []
        for mwe in self.vMWEs:
            if mwe.isLeftEmbedded:
                res.append(mwe)
        return res

    def getRecognizableVMWEs(self):
        res = []
        for mwe in self.vMWEs:
            if mwe.isRecognizable:
                res.append(mwe)
        return res

    def setTokenParent(self):
        for token in self.tokens:
            if token.parentMWEs:
                parents = filterNonRecognizableVMWEs(list(token.parentMWEs))
                if parents:
                    parents = sorted(parents, key=lambda parent: (len(parent)))
                    token.parent = parents
                else:
                    token.parent = None
            else:
                token.parent = None

    def __str__(self):

        vMWEText, identifiedMWE = '', ''
        if self.vMWEs:
            vMWEText += 'MWEs:\n'
            for vMWE in self.vMWEs:
                vMWEText += str(vMWE) + '\n'
        if self.identifiedVMWEs:
            identifiedMWE = ' Identified MWEs: \n'
            for mwe in self.identifiedVMWEs:
                identifiedMWE += str(mwe) + '\n'

        transStr = ''
        trans = self.initialTransition
        while trans:
            transStr += str(trans)
            trans = trans.next
        if self.vMWEs:
            text = ''
            for token in self.tokens:
                if token.parentMWEs is not None and len(token.parentMWEs) > 0:
                    idxs = ''
                    for vmwe in token.parentMWEs:
                        idxs += ':' + str(vmwe.id) if idxs else str(vmwe.id)
                    text += idxs + ' **' + token.text + '**' + ' '
                else:
                    text += token.text + ' '
        else:
            text = self.text
        return '{0}- {1}:: {2}\n{3}{4}{5}'.format(self.id, self.sentid, text, vMWEText, identifiedMWE, transStr)

    def __iter__(self):
        for vmwe in self.vMWEs:
            yield vmwe


class VMWE:
    """
        A class used to encapsulate the information of a verbal multi-word expression
    """

    def __init__(self, idx, tokens=None, type='', isEmbedded=False, isEmbedder=False, isLeftEmbedded=False,
                 isLeftEmbedder=False, isRightEmbedded=False, isRightEmbedder=False, isInterleaving=False,
                 isContinous=False, isRecognizable=True, isMiddleEmbedded=False, isMiddleEmbedder=False, parent=None):
        self.id = int(idx)
        self.tokens = tokens if tokens else []
        self.type = type
        self.isEmbedded = isEmbedded
        self.isEmbedder = isEmbedder
        self.isLeftEmbedded = isLeftEmbedded
        self.isRightEmbedded = isRightEmbedded
        self.isLeftEmbedder = isLeftEmbedder
        self.isRightEmbedder = isRightEmbedder
        self.isMiddleEmbedded = isMiddleEmbedded
        self.isMiddleEmbedder = isMiddleEmbedder
        self.isInterleaving = isInterleaving
        self.isContinous = isContinous
        self.directParent = None
        self.parent = parent
        self.parsedByOracle = False
        self.isInterleaver = False
        self.isBadAnnotated = False
        self.isAletrnating = False
        self.isRecognizable = isRecognizable

    def getId(self):
        return self.id

    def isAttachedVMWE(self):
        if len(self.tokens) == 1:
            return True
        idxs = []
        for token in self.tokens:
            idxs.append(token.position)
        isContinous = True
        for i in xrange(min(idxs), max(idxs) + 1):
            if i not in idxs:
                isContinous = False
        return isContinous

    def __str__(self):
        return '{0}- {1}: {2} {3}'.format(self.id, self.type, self.getLemmaString(), self.getCaracter())

    def getString(self):
        result = ''
        for token in self.tokens:
            result += token.text + ' '
        return result[:-1].lower()

    def getLemmaString(self):
        result = ''
        for token in self.tokens:
            if token.lemma.strip():
                result += token.lemma + ' '
            else:
                result += token.text + ' '
        return result[:-1].lower()

    def getTokenPositionString(self):
        result = '.'
        for token in self.tokens:
            result += token.getPositionString()
        return result

    def In(self, vmwes):

        for vmwe in vmwes:
            if vmwe.getString() == self.getString():
                return True

        return False

    def getCaracter(self):
        res = '(+)' if self.isRecognizable else '(-)'
        if self.isInterleaving:
            res += ':Interleaving'
        if self.isEmbedder:
            res += ':Embedder'
        if self.isEmbedded:
            if self.isLeftEmbedded:
                res += ':Left Embedded'
            elif self.isRightEmbedded:
                res += ':Right Embedded'
            elif self.isMiddleEmbedded:
                res += ':Middle Embedded'
            elif self.isEmbedded:
                res += 'Dist Embedded'
        return res

    def __iter__(self):
        for t in self.tokens:
            yield t

    def __eq__(self, other):
        if self.getTokenPositionString() == other.getTokenPositionString():
            return True
        return False

    def __hash__(self):
        return hash(self.getTokenPositionString())

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, vmwe):
        if not isinstance(vmwe, VMWE):
            raise TypeError()
        if vmwe is self or vmwe.getTokenPositionString() == self.getTokenPositionString():
            return False
        if vmwe.getTokenPositionString() in self.getTokenPositionString():
            return True
        for token in vmwe.tokens:
            if token.getPositionString() not in self.getTokenPositionString():
                return False
        return True


class Token:
    """
        a class used to encapsulate all the information of a sentence tokens
    """

    def __init__(self, position, txt, lemma='', posTag='', abstractPosTag='', morphologicalInfo=None,
                 dependencyParent=-1,
                 dependencyLabel=''):
        self.position = int(position)
        self.text = txt
        self.lemma = lemma
        self.abstractPosTag = abstractPosTag
        self.posTag = posTag
        if not morphologicalInfo:
            self.morphologicalInfo = []
        else:
            self.morphologicalInfo = morphologicalInfo
        self.dependencyParent = dependencyParent
        self.dependencyLabel = dependencyLabel
        self.parentMWEs = []
        self.directParent = None
        self.parent = None

    def setParent(self, vMWE):
        self.parentMWEs.append(vMWE)

    def getLemma(self):
        if self.lemma:
            return self.lemma.strip().lower()
        return self.text.strip().lower()

    def getPositionString(self):
        if self.position < 10:
            return '0' + str(self.position) + '.'
        else:
            return str(self.position) + '.'

    def getDirectParent(self):
        if not self.parentMWEs:
            return None
        self.directParent = None
        if self.parentMWEs:
            if len(self.parentMWEs) == 1:
                self.directParent = self.parentMWEs[0]
            else:
                parents = sorted(self.parentMWEs,
                                 key=lambda mwe: (mwe.isInterleaving, mwe.isEmbedded, len(mwe)),
                                 reverse=True)
                for parent in parents:
                    if not parent.isInterleaving:
                        self.directParent = parent
                        break
        return self.directParent

    def In(self, vmwe):
        for token in vmwe.tokens:
            if token.text.lower() == self.text.lower() and token.position == self.position:
                return True
        return False

    def isMWT(self):
        if self.parentMWEs:
            for vmw in self.parentMWEs:
                if len(vmw.tokens) == 1:
                    return vmw
        return None

    def getTokenOrLemma(self):
        useLemma = configuration["model"]["embedding"]["lemma"]
        if not useLemma:
            return self.text.lower()
        if self.lemma:
            return self.lemma.lower()
        return self.text.lower()

    def getStandardKey(self, getPos=False, getToken=False):
        if getPos:
            return self.posTag.lower()
        if getToken:
            return self.getTokenOrLemma()
        return self.getTokenOrLemma() + '_' + self.posTag.lower()

    def __str__(self):
        parentTxt = ' '
        if self.parentMWEs:
            for par in self.parentMWEs:
                if parentTxt:
                    parentTxt += '; ' + str(par.id)
                else:
                    parentTxt += str(par.id)
        return str(self.position) + ' : ' + self.text + ' : ' + self.posTag + parentTxt + '\n'


def saveObj(obj, name):
    with open(os.path.join(configuration["path"]["projectPath"], 'ressources/' + name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def loadObj(name):
    with open(os.path.join(configuration["path"]["projectPath"], 'ressources/' + name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def readConlluFile(conlluFile):
    sentences = []
    with open(conlluFile, 'r') as corpusFile:
        sent, senIdx, sentId = None, 0, ''
        lineNum, missedUnTag, missedExTag = 0, 0, 0
        for line in corpusFile:
            if len(line) > 0 and line.endswith('\n'):
                line = line[:-1]
            if line.startswith('# sentid:'):
                sentId = line.split('# sentid:')[1].strip()
            elif line.startswith('# sentence-text:'):
                continue

            elif line.startswith('1\t'):
                if sentId.strip():
                    sent = Sentence(senIdx, sentid=sentId)
                else:
                    sent = Sentence(senIdx)
                senIdx += 1
                sentences.append(sent)

            if not line.startswith('#'):
                lineParts = line.split('\t')

                if len(lineParts) != 10 or '-' in lineParts[0]:
                    continue

                lineNum += 1
                if lineParts[3] == '_':
                    missedUnTag += 1
                if lineParts[4] == '_':
                    missedExTag += 1

                morpho = ''
                if lineParts[5] != '_':
                    morpho = lineParts[5].split('|')
                if lineParts[6] != '_':
                    token = Token(lineParts[0], lineParts[1].lower(), lemma=lineParts[2],
                                  abstractPosTag=lineParts[3], morphologicalInfo=morpho,
                                  dependencyParent=int(lineParts[6]),
                                  dependencyLabel=lineParts[7])
                else:
                    token = Token(lineParts[0], lineParts[1].lower(), lemma=lineParts[2],
                                  abstractPosTag=lineParts[3], morphologicalInfo=morpho,
                                  dependencyLabel=lineParts[7])
                useUniversalPOS = configuration["preprocessing"]["data"]["universalPOS"]
                if useUniversalPOS:
                    token.posTag = lineParts[3]
                else:
                    if lineParts[4] != '_':
                        token.posTag = lineParts[4]
                    else:
                        token.posTag = lineParts[3]
                # Associate the token with the sentence
                sent.tokens.append(token)
                sent.text += token.text + ' '
    return sentences


def getTrainAndTestConlluPath(path):
    testFiles = configuration["files"]["test"]
    trainFiles = configuration["files"]["train"]
    if os.path.isfile(
            os.path.join(path, trainFiles["depAuto"])):
        conlluFile = os.path.join(path, trainFiles["depAuto"])
        testConllu = os.path.join(path, testFiles["depAuto"])
    elif os.path.isfile(os.path.join(path, trainFiles["posAuto"])):
        conlluFile = os.path.join(path, trainFiles["posAuto"])
        testConllu = os.path.join(path, testFiles["posAuto"])
    elif os.path.isfile(os.path.join(path, trainFiles["conllu"])):
        conlluFile = os.path.join(path, trainFiles["conllu"])
        testConllu = os.path.join(path, testFiles["conllu"])
    else:
        conlluFile, testConllu = None, None
    if conlluFile and testConllu:
        sys.stdout.write('# Train file = {0}\n'.format(conlluFile.split('/')[-1]))
        sys.stdout.write('# Test file = {0}\n'.format(testConllu.split('/')[-1]))
    return conlluFile, testConllu


def readMWEFile(mweFile):
    sentences = []
    with open(mweFile) as corpusFile:
        # Read the corpus file
        sent = None
        senIdx = 1
        for line in corpusFile:
            if len(line) > 0 and line.endswith('\n'):
                line = line[:-1]
            if line.startswith('1\t'):
                # sentId = line.split('# sentid:')[1]
                if sent:
                    sent.setTextandPOS()
                    sent.recognizeEmbedded()
                    sent.recognizeInterleaving()
                sent = Sentence(senIdx)
                senIdx += 1
                sentences.append(sent)

            elif line.startswith('# sentence-text:'):
                if len(line.split(':')) > 1:
                    sent.text = line.split('# sentence-text:')[1]

            lineParts = line.split('\t')

            # Empty line or lines of the form: "8-9	can't	_	_"
            if len(lineParts) != 4 or '-' in lineParts[0]:
                continue
            token = Token(lineParts[0], lineParts[1])
            # Trait the MWE
            # if not forTest and lineParts[3] != '_':
            if lineParts[3] != '_':
                vMWEids = lineParts[3].split(';')
                for vMWEid in vMWEids:
                    idx = int(vMWEid.split(':')[0])
                    # New MWE captured
                    if idx not in sent.getWMWEIds():
                        type = str(vMWEid.split(':')[1])
                        vMWE = VMWE(idx, [token], type)
                        sent.vMWEs.append(vMWE)
                    # Another token of an under-processing MWE
                    else:
                        vMWE = sent.getVMWE(idx)
                        if vMWE is not None:
                            vMWE.tokens.append(token)
                    # associate the token with the MWE
                    token.setParent(vMWE)
            # Associate the token with the sentence
            sent.tokens.append(token)
        return sentences


def integrateMweFile(mweFile, sentences):
    mweNum = 0
    with open(mweFile) as corpusFile:
        # Read the corpus file
        lines = corpusFile.readlines()
        noSentToAssign = False
        sentIdx = 0
        for line in lines:
            if line == '\n' or line.startswith('# sentence-text:') or (
                    line.startswith('# sentid:') and noSentToAssign):
                continue
            if len(line) > 0 and line.endswith('\n'):
                line = line[:-1]
            if line.startswith('1\t'):
                sent = sentences[sentIdx]
                sentIdx += 1
            lineParts = line.split('\t')
            if '-' in lineParts[0]:
                continue
            if lineParts and len(lineParts) == 4 and lineParts[3] != '_':

                token = sent.tokens[int(lineParts[0]) - 1]
                vMWEids = lineParts[3].split(';')
                for vMWEid in vMWEids:
                    idx = int(vMWEid.split(':')[0])
                    # New MWE captured
                    if idx not in sent.getWMWEIds():
                        if len(vMWEid.split(':')) > 1:
                            type = str(vMWEid.split(':')[1])
                            vMWE = VMWE(idx, [token], type)
                        else:
                            vMWE = VMWE(idx, [token])
                        mweNum += 1
                        sent.vMWEs.append(vMWE)
                    # Another token of an under-processing MWE
                    else:
                        vMWE = sent.getVMWE(idx)
                        if vMWE:
                            vMWE.tokens.append(token)
                    # associate the token with the MWE
                    token.setParent(vMWE)

    return mweNum


def getVMWESents(sents, num):
    result, idx = [], 0
    evalConfig = configuration["evaluation"]
    if evalConfig["debug"]:
        # if settings.CORPUS_SHUFFLE:
        #     shuffle(sents)
        for sent in sents:
            # if sent.vMWEs:
            result.append(sent)
            idx += 1
            if idx >= num:
                return result
        if len(result) < num:
            for sent in sents:
                if sent not in result:
                    result.append(sent)
                if len(result) >= num:
                    return result
        return result
    # result, idx = [], 0
    return sents[:num]
    # if settings.CORPUS_SHUFFLE:
    #     shuffle(sents)
    # for sent in sents:
    #     # if sent.vMWEs:
    #     result.append(sent)
    #     idx += 1
    #     if idx >= num:
    #         return result
    # if len(result) < num:
    #     for sent in sents:
    #         if sent not in result:
    #             result.append(sent)
    #         if len(result) >= num:
    #             return result
    # return result


def getTokens(elemlist):
    if str(elemlist.__class__) == 'corpus.Token':  # isinstance(elemlist, Token):
        return [elemlist]
    if isinstance(elemlist, list):
        result = []
        for elem in elemlist:
            if str(elem.__class__) == 'corpus.Token':
                result.append(elem)
            elif isinstance(elem, list) and len(elem) == 1 and isinstance(elem[0], list):
                result.extend(getTokens(elem[0]))
            elif isinstance(elem, list) and len(elem):
                result.extend(getTokens(elem))
        return result
    return [elemlist]


def getTokenLemmas(tokens):
    text = ''
    tokens = getTokens(tokens)
    for token in tokens:
        if token.lemma != '':
            text += token.lemma + ' '
        else:
            text += token.text + ' '
    return text.strip()


def getParent(tokens):
    if hasOrphanToken(tokens):
        return None
    parent = tokens[0].parent
    for token in tokens:
        if token.parent != parent:
            return None
    return parent


def getParents(tokens, allChildren=True):
    if hasOrphanToken(tokens):
        return []
    parents = regroupParents(tokens)
    parents = filterNonRecognizableVMWEs(parents)
    if allChildren:
        parents = filterOnLength(parents, len(tokens))
    parents = filterPartialParents(parents, tokens)
    parents = sorted(parents, key=lambda parent: (len(parent)))
    return parents


def filterPartialParents(parents, childs):
    for parent in list(parents):
        for child in childs:
            if parent not in child.parentMWEs:
                if parent in parents:
                    parents.remove(parent)
    return parents


def filterOnLength(mwes, length):
    for mwe in list(mwes):
        if len(mwe) != length:
            mwes.remove(mwe)
    return mwes


def filterNonRecognizableVMWEs(mwes):
    for mwe in list(mwes):
        if not mwe.isRecognizable:
            mwes.remove(mwe)
    return mwes


def isOrphanToken(token):
    if token and token.parentMWEs:
        for vmwe in token.parentMWEs:
            if vmwe.isRecognizable:
                return False
        return True
    if token and (not token.parentMWEs):
        return True


def hasOrphanToken(tokens):
    # Do they have all a parent?
    for token in tokens:
        if isOrphanToken(token):
            return True
    return False


def regroupParents(tokens):
    parents = set()
    for token in tokens:
        for parent in token.parentMWEs:
            parents.add(parent)
    return list(parents)


def getVMWEByTokens(tokens):
    vmwes = regroupParents(tokens)
    vmwes = filterOnLength(vmwes, len(tokens))
    if vmwes:
        return vmwes[0]
    return None
