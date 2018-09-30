#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import datetime
import os
import sys
from collections import Counter
from random import shuffle
from random import uniform

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import reports
from config import configuration
from corpus import getTokens
from reports import seperator, doubleSep, tabs

device = "cpu"
dtype = torch.float

enableCategorization = False

unk = configuration['constants']['unk']
empty = configuration['constants']['empty']
number = configuration['constants']['number']


class TransitionClassifier(nn.Module):
    """
    version avec embeddings / lstm sur tous les mots de la phrase, et calcul de perte sur
    toutes les transitions de la phrase.
    """

    def __init__(self, corpus):
        """
        if use_pretrained_w_emb, the indices.w_embeddings_matrix will be used
        """
        super(TransitionClassifier, self).__init__()
        self.tokenVocab, self.posVocab = getVocab(corpus)
        kiperConf = configuration['kiperwasser']
        self.p_embeddings = nn.Embedding(len(self.posVocab), kiperConf['posDim'])
        self.w_embeddings = nn.Embedding(len(self.tokenVocab), kiperConf['wordDim'])
        embeddingDim = kiperConf['wordDim'] + kiperConf['posDim']
        self.lstm = nn.LSTM(embeddingDim,
                            kiperConf['lstmUnitNum'],
                            bidirectional=True,
                            num_layers=kiperConf['lstmLayerNum'],
                            dropout=kiperConf['lstmDropout'] if kiperConf['lstmLayerNum'] > 1 else 0)
        # init hidden and cell states h0 c0
        self.hiddenLstm = initHiddenLstm()
        # * 2 because bidirectional
        self.linear1 = nn.Linear(kiperConf['focusedElemNum'] * kiperConf['lstmUnitNum'] * 2, kiperConf['dense1'])
        # dropout here is very detrimental
        # self.dropout1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(kiperConf['dense1'], 8 if enableCategorization else 4)

    def forward(self, sentEmbs, activeElemIdxs):
        """ propagation avant, pour prediction d"une transition
        input =
        - embeddings as output from compute_contextualized_embeddings
        (contextualized with lstm layer)
        - the list of the lidxs of the nodes to focus on, according to current configuration
        """
        kiperConf = configuration['kiperwasser']
        # select the contextualized embeddings for the focused_elements (given configuration)
        activeElems = selectRows(sentEmbs, activeElemIdxs).view((1, -1))
        out = f.relu(self.linear1(activeElems)) if kiperConf['denseActivation'] == 'relu' else f.tanh(
            self.linear1(activeElems))
        if kiperConf['denseDropout']:
            out = self.dropout1(out)
        out = self.linear2(out)
        scores = f.log_softmax(out, dim=1)
        return scores

    def predict(self, trans, sentEmbs):
        """
        prediction of score vector (for each transition)
        Returns sorted score list / corresponding class id list
        """
        activeElemIdxs = getFocusedElems(trans.configuration)
        # cet appel comprend l"appel de self.forward
        scores = self.forward(sentEmbs, activeElemIdxs)
        # on obtient par ex, si 3 transitions: tensor([[-1.3893, -1.6119, -0.5956]])
        # (cf. minibatch de 1)
        _, sortedIndices = torch.sort(scores[0], descending=True)
        # return sorted_scores, sorted_indices
        return sortedIndices

    def getContextualizedEmbs(self, tokenIdxs, posIdxs):
        """
        input = sequence of word ids / sequence of pos ids of the sentence
        as output by get_sequence_feats_for_whole_sentence)

        (each are a LongTensor( list of ids )
        and contain as first element the not_existing_node, and the ROOT as second element,
        then all positions of the sentence

        output = tensor of shape (sequence length, self.embedding_dim)
        with lstm-contextualized vectors for each position, computed over non-contextualized embeddings
        """
        tokenEmbed = self.w_embeddings(tokenIdxs).to(device)
        posEmbed = self.p_embeddings(posIdxs).to(device)
        sentEmbed = torch.cat([tokenEmbed, posEmbed], 1).to(device)
        self.hiddenLstm = initHiddenLstm()
        lstmHiddenSeq, self.hiddenLstm = self.lstm(sentEmbed.view(len(tokenIdxs), 1, -1), self.hiddenLstm)
        return lstmHiddenSeq.view(len(tokenIdxs), -1)

from random import randint
def train(corpus, trainedModel=None, trainValidation=False, fileNum=0):
    """
    version avec bi-LSTM sur toute la phrase, et mise à jour des paramètres à chaque phrase
    (calcul de la perte pour une phrase complete)
    """
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kiperConf = configuration['kiperwasser']
    # nb of sentence positions taken to build input vector representing a parse configuration
    model = TransitionClassifier(corpus).to(device) if not trainValidation else trainedModel
    optimizer = getOptimizer(model.parameters())
    lossFunction = nn.NLLLoss()
    if not trainValidation:
        fileNum = randint(0, 500)
    filePath = os.path.join(configuration['path']['projectPath'], 'tmp', str(fileNum) + '.' + kiperConf['file'])
    sys.stdout.write('\n' + filePath + '\n')
    # losses of validation set for each epoch
    epochLosses, validLosses, validAccuracies = [], [], []
    # if kiperConf['verbose']:
    sys.stderr.write(reports.tabs + str(model) + reports.doubleSep if kiperConf['verbose'] else '')
    sys.stderr.write(reports.tabs + str(optimizer) + reports.doubleSep if kiperConf['verbose'] else '')
    sys.stderr.write(reports.tabs + str(lossFunction) + reports.doubleSep if kiperConf['verbose'] else '')
    # ------------ if validation asked -------------
    validSeuil = configuration['model']['train']['validationSplit']
    pointer = int(len(corpus.trainingSents) * (1 - validSeuil))
    validSents = corpus.trainingSents[pointer:]
    trainSents = corpus.trainingSents[:pointer] if not trainValidation else validSents
    for epoch in range(kiperConf['epochs']):
        sys.stderr.write(reports.tabs + "Epoch %d....\n" % epoch if kiperConf['verbose'] else '')
        start, epochLoss, usedSents = datetime.datetime.now(), 0, 0
        # shuffle sentences
        sentRanks = range(len(trainSents))
        shuffle(sentRanks)
        for i in sentRanks:
            sent = trainSents[i]
            model.zero_grad()
            sentLoss = getSentLoss(sent, model, lossFunction)
            if sentLoss:
                usedSents += 1
                epochLoss += sentLoss.item()
                sentLoss.backward()
                optimizer.step()
        epochLosses.append(epochLoss)
        # if kiperConf['verbose']:
        sys.stderr.write("Number of used sentences in train = %d\n" % usedSents if kiperConf['verbose'] else '')
        sys.stderr.write("Total loss for epoch %d: %f\n" % (epoch, epochLoss) if kiperConf['verbose'] else '')
        if not trainValidation:
            valLoss = getCorpusLoss(validSents, model, lossFunction)
            validAcc = evaluate(validSents, model)
            sys.stdout.write('validAcc: ' + str(validAcc) + '\n')
            # if kiperConf['verbose']:
            sys.stderr.write("validation loss after epoch %d : %f\n" % (epoch, valLoss) if kiperConf['verbose'] else '')
            # save model if validation loss has decreased
            # if validLosses and valLoss <= validLosses[-1]:
            if validAccuracies and validAcc and validAcc > max(validAccuracies):
                torch.save(model, filePath)
            # early stopping
            elif validAccuracies and validAcc and kiperConf['earlyStop'] and validAcc <= max(validAccuracies):
                model = torch.load(filePath)
                # validLosses and (valLoss >= validLosses[-1]):
                # sys.stderr.write("validation loss has increased (), stop and retrain using %d epochs\n" % epoch)
                # sys.stderr.write("validation loss has increased (), stop and retrain using %d epochs\n" % epoch)
                sys.stderr.write("identification accuracy has decreased (), stop and retrain using %d epochs\n" % (epoch - 1))
                if validSeuil:
                    kiperConf['epochs'] = epoch - 1  # (cf. epoch est décalé de 1)
                    # return train(corpus, model, trainValidation=True,fileNum=fileNum)
            # if no validation set: save iff loss on training set decreases
            # else:
            #     if len(epochLosses) > 1 and epochLoss < epochLosses[-2]:
            #         torch.save(model, filePath)
            validLosses.append(valLoss)
            validAccuracies.append(validAcc)
        sys.stdout.write("Epoch has taken {0}\n".format(datetime.datetime.now() - start)
                         if kiperConf['verbose'] else '')
    if not trainValidation:
        return train(corpus, model, trainValidation=True)

    return model


from parser import parse
import evaluation


def evaluate(sents, model):
    parse(sents, model)
    return evaluation.evaluate(sents, loggingg=False)[1]


def initHiddenLstm():
    """
    Before we"ve done anything, we dont have any hidden state.
    The axes semantics are (num_layers, minibatch_size, hidden_dim)
    h_0 is of shape (num_layers * num_directions, batch, hidden_size):
                    tensor containing the initial hidden state for each element in the batch
    same for c_0 = initial cell state
    here num_directions = 2 (bidirectional), batch = 1
    :return:
    """
    return torch.zeros(configuration['kiperwasser']['lstmLayerNum'] * 2, configuration['kiperwasser']['batch'],
                       configuration['kiperwasser']['lstmUnitNum']).to(device), \
           torch.zeros(configuration['kiperwasser']['lstmLayerNum'] * 2, configuration['kiperwasser']['batch'],
                       configuration['kiperwasser']['lstmUnitNum']).to(device)
    # return (torch.zeros(lstmLayerNum * 2, 1, lstmUnitNum),
    #        torch.zeros(lstmLayerNum * 2, 1, lstmUnitNum))


def selectRows(sentEmbeds, idxs):
    """
    extract given rows (= axis 0) from a given 2 dim tensor
    """
    # tuplee = [tensor[idx, :].view(1, -1) for idx in idxs]
    # return torch.cat(tuplee)
    results = []
    for idx in idxs:
        if idx == -1:
            results.append(torch.zeros((1, 2 * configuration['kiperwasser']['lstmUnitNum']), dtype=dtype).to(device))
        else:
            results.append(sentEmbeds[idx - 1].view(1, -1).to(device))
    return torch.cat(results)


def getCorpusLoss(sents, model, lossFunction):
    """
    returns the loss for a list of gold dgs
    """
    loss = 0
    for sent in sents:
        sentLoss = getSentLoss(sent, model, lossFunction)
        loss += sentLoss.item() if sentLoss else 0.
    return loss


def getIdxs(sent, model):
    tokenIdxs, POSIdxs = [], []
    for token in sent.tokens:
        isDigit = False
        for c in token.getTokenOrLemma():
            if c.isdigit():
                isDigit = True
        if token.getTokenOrLemma() in model.tokenVocab:
            tokenIdxs.append(model.tokenVocab[token.getTokenOrLemma()])
        elif isDigit:
            tokenIdxs.append(model.tokenVocab[number])
        else:
            tokenIdxs.append(model.tokenVocab[unk])

        if token.posTag.lower() in model.posVocab:
            POSIdxs.append(model.posVocab[token.posTag.lower()])
        else:
            POSIdxs.append(model.posVocab[unk])
    return torch.LongTensor(tokenIdxs).to(device), torch.LongTensor(POSIdxs).to(device)


def getVocab(corpus):
    tokenCounter, posCounter = Counter(), Counter()
    for s in corpus.trainingSents:
        for t in s.tokens:
            tokenCounter.update({t.getTokenOrLemma(): 1})
            posCounter.update({t.posTag.lower(): 1})
    if configuration['model']['embedding']['compactVocab']:
        for t in tokenCounter.keys():
            if t not in corpus.mweTokenDictionary:
                del tokenCounter[t]
    else:
        for t in tokenCounter.keys():
            if tokenCounter[t] == 1 and uniform(0, 1) < configuration['constants']['alpha']:
                del tokenCounter[t]
    tokenCounter.update({unk: 1, number: 1})
    posCounter.update({unk: 1})
    printVocabReport(tokenCounter, posCounter)
    return {w: i for i, w in enumerate(tokenCounter.keys())}, {w: i for i, w in enumerate(posCounter.keys())}


def printVocabReport(tokenCounter, posCounter):
    res = seperator + tabs + "Vocabulary" + doubleSep
    res += tabs + "Tokens := {0} * POS : {1}".format(len(tokenCounter), len(posCounter)) \
        if not configuration['xp']['compo'] else ""
    res += seperator
    return res


def getFocusedElems(config):
    idxs = []
    if config.stack and len(config.stack) > 1:
        for t in getTokens(config.stack[-2])[:2]:
            idxs.append(t.position)
    while len(idxs) < 2:
        idxs = idxs + [-1]

    if config.stack:
        for t in getTokens(config.stack[-1])[:4]:
            idxs.append(t.position)
    while len(idxs) < 6:
        idxs = idxs + [-1]

    if config.buffer:
        for t in config.buffer[:2]:
            idxs.append(t.position)

    while len(idxs) < 8:
        idxs = idxs + [-1]

    return idxs


def getSentLoss(sent, model, lossFunction):
    """
    returns the whole loss for the sentence in gold_dg
    (return type is that of Torch.nn losses: differentiable Tensor of size 1)

    unless no gold transition sequence could be found
    (e.g. if gold_dg is a non proj tree and the transition set does not handle it)
    in which case returns None
    """
    # list of losses after each decision (transition) for the sentence
    transLosses = []
    tokenIdxs, posIdxs = getIdxs(sent, model)
    sentEmb = model.getContextualizedEmbs(tokenIdxs, posIdxs)

    trans = sent.initialTransition

    # transFeedNum, transRepetitionTaux = 0, 1
    while trans and trans.next:
        # Used to reduce the number of used transitions by replacing all Mark as transition by Mark as OTH
        goldT = 3 if trans.next.type.value > 2 and not enableCategorization else trans.next.type.value
        focusedIdxs = getFocusedElems(trans.configuration)
        # prediction
        predT = model.forward(sentEmb, focusedIdxs)
        # _, sortedIndices = torch.sort(y_pred_vec[0], descending=True)
        # loss
        # if y != sortedIndices[0]:
        #    loss = 1
        #    transLosses.append(loss)
        loss = lossFunction(predT, toTensor(goldT))
        transLosses.append(loss)
        # epoch_loss += loss.data
        # if trans.isImportantTrans():
        #     transFeedNum += 1
        #     if transFeedNum == transRepetitionTaux:
        #         transFeedNum = 0
        #         trans = trans.next
        # else:
        #    trans = trans.next
        trans = trans.next
    return sum(transLosses)


def getOptimizer(params):
    sys.stdout.write("# Network optimizer = Adagrad, learning rate = {0}\n".format(configuration['kiperwasser']['lr']))
    return optim.Adagrad(params, lr=configuration['kiperwasser']['lr'])


def toTensor(label):
    """
    cree un Tensor de shape (1,); i.e. on crée le label gold pour un mini-batch de 1
    """
    return torch.LongTensor([label]).to(device)
