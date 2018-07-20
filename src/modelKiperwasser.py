#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import reports
from config import configuration
from corpus import getTokens
from transitions import TransitionType
from wordEmbLoader import unk

lstmUnitNum = 32
focusedElemNum = 8
lstmLayerNum = 2

w_vocab_size = 0
p_vocab_size = 0
device = 'cpu'
dtype = torch.float


class TransitionClassifier(nn.Module):
    """
    version avec embeddings / lstm sur tous les mots de la phrase, et calcul de perte sur
    toutes les transitions de la phrase.
    """

    def __init__(self):
        """
        if use_pretrained_w_emb, the indices.w_embeddings_matrix will be used
        """
        super(TransitionClassifier, self).__init__()
        embConf = configuration["model"]["embedding"]
        self.p_embeddings = nn.Embedding(p_vocab_size, embConf["posEmb"])
        self.w_embeddings = nn.Embedding(w_vocab_size, embConf["tokenEmb"])
        embeddingDim = embConf["tokenEmb"] + embConf["posEmb"]
        self.lstm = nn.LSTM(embeddingDim,
                            lstmUnitNum,
                            bidirectional=True,
                            num_layers=lstmLayerNum,
                            dropout=0.5)
        # init hidden and cell states h0 c0
        self.lstm_hidden_and_cell = init_lstm_hidden_and_cell()
        # * 2 because bidirectional
        self.linear1 = nn.Linear(focusedElemNum * lstmUnitNum * 2, 128)
        # dropout here is very detrimental
        self.dropout1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(128, len(TransitionType))

    def forward(self, sentEmbs, activeElemIdxs):
        """ propagation avant, pour prediction d'une transition
        input =
        - embeddings as output from compute_contextualized_embeddings
        (contextualized with lstm layer)

        - the list of the lidxs of the nodes to focus on, according to current configuration
        """
        # select the contextualized embeddings for the focused_elements (given configuration)
        activeElems = selectRows(sentEmbs, activeElemIdxs).view((1, -1))
        out = f.relu(self.linear1(activeElems))
        out = self.dropout1(out)
        out = self.linear2(out)
        scores = f.log_softmax(out, dim=1)
        return scores

    def predict(self, sentEmbs, trans):
        """
        prediction of score vector (for each transition)
        Returns sorted score list / corresponding class id list
        """
        activeElemIdxs = getFocusedElems(trans.configuration)
        # cet appel comprend l'appel de self.forward
        scores = self(sentEmbs, activeElemIdxs)
        # on obtient par ex, si 3 transitions: tensor([[-1.3893, -1.6119, -0.5956]])
        # (cf. minibatch de 1)
        _, sortedIndices = torch.sort(scores[0], descending=True)
        # return sorted_scores, sorted_indices
        return sortedIndices[0]

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
        sent_length = len(posIdxs)
        sentence_w_embeddings = self.w_embeddings(tokenIdxs).to(device)
        sentence_p_embeddings = self.p_embeddings(posIdxs).to(device)
        sentence_embeddings = torch.cat([sentence_w_embeddings, sentence_p_embeddings], 1).to(device)
        lstm_hidden_seq, self.lstm_hidden_and_cell = \
            self.lstm(sentence_embeddings.view(sent_length, 1, -1), self.lstm_hidden_and_cell)#.to(device)
        return lstm_hidden_seq.view(sent_length, -1)


def init_lstm_hidden_and_cell():
    """
    Before we've done anything, we dont have any hidden state.
    The axes semantics are (num_layers, minibatch_size, hidden_dim)
    h_0 is of shape (num_layers * num_directions, batch, hidden_size):
                    tensor containing the initial hidden state for each element in the batch
    same for c_0 = initial cell state
    here num_directions = 2 (bidirectional), batch = 1
    :return:
    """
    return torch.zeros(lstmLayerNum * 2, 1, lstmUnitNum), torch.zeros(lstmLayerNum * 2, 1, lstmUnitNum)
    # return (torch.zeros(lstmLayerNum * 2, 1, lstmUnitNum),
    #        torch.zeros(lstmLayerNum * 2, 1, lstmUnitNum))


def selectRows(tensor, idxs):
    """
    extract given rows (= axis 0) from a given 2 dim tensor
    """
    # tuplee = [tensor[idx, :].view(1, -1) for idx in idxs]
    # return torch.cat(tuplee)
    results = []
    for idx in idxs:
        if idx != -1:
            results.append(torch.zeros((1, 2 * lstmUnitNum), dtype=torch.float))
        else:
            results.append(tensor[idx, :].view(1, -1))
    return torch.cat(results)


def getCorpusLoss(sents, normalizer, model, loss_function):  # , device):
    """
    returns the loss for a list of gold dgs
    """
    loss = 0
    for sent in sents:
        sentLoss = getSentLoss(sent, normalizer, model, loss_function)  # , device)
        loss += sentLoss.item() if sentLoss else 0.  # transform into float
    return loss


def getIdxs(sent, normalizer):
    tokenIdxs, POSIdxs = [], []
    for token in sent.tokens:
        if token.getTokenOrLemma() in normalizer.vocabulary.tokenIndices:
            tokenIdxs.append(normalizer.vocabulary.tokenIndices[token.getTokenOrLemma()])
        else:
            tokenIdxs.append(normalizer.vocabulary.tokenIndices[unk])

        if token.posTag.lower() in normalizer.vocabulary.posIndices:
            POSIdxs.append(normalizer.vocabulary.posIndices[token.posTag.lower()])
        else:
            POSIdxs.append(normalizer.vocabulary.posIndices[unk])
    return torch.LongTensor(tokenIdxs).to(device), torch.LongTensor(POSIdxs).to(device)


def getFocusedElems(config):
    idxs = []
    if config.stack and len(config.stack) > 1:
        for t in getTokens(config.stack[-2])[:2]:
            idxs.append(t.position)
    while len(idxs) < 2:
        idxs = [-1] + idxs

    if config.stack:
        for t in getTokens(config.stack[-1])[:4]:
            idxs.append(t.position)
    while len(idxs) < 6:
        idxs = [-1] + idxs

    if config.buffer:
        for t in config.buffer[:2]:
            idxs.append(t.position)

    while len(idxs) < 8:
        idxs = [-1] + idxs

    return idxs


def getSentLoss(sent, normalizer, model, loss_function):  # , device):
    """
    returns the whole loss for the sentence in gold_dg
    (return type is that of Torch.nn losses: differentiable Tensor of size 1)

    unless no gold transition sequence could be found
    (e.g. if gold_dg is a non proj tree and the transition set does not handle it)
    in which case returns None
    """
    # list of losses after each decision (transition) for the sentence
    sentence_losses = []
    tokenIdxs, posIdxs = getIdxs(sent, normalizer)
    sentEmb = model.getContextualizedEmbs(tokenIdxs, posIdxs)
    trans = sent.initialTransition
    while trans and trans.next:
        y = trans.next.type.value
        # print("gold trans %s is to apply on config: %s" % (t, c))
        focused_lidxs = getFocusedElems(trans.configuration)
        # prediction
        y_pred_vec = model(sentEmb, focused_lidxs)
        # loss
        loss = loss_function(y_pred_vec, make_y(y))
        sentence_losses.append(loss)
        # epoch_loss += loss.data
        trans = trans.next
    return sum(sentence_losses)


def train(corpus, normalizer):
    """
    version avec bi-LSTM sur toute la phrase, et mise à jour des paramètres à chaque phrase
    (calcul de la perte pour une phrase complete)
    """
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainConf = configuration["model"]["train"]
    global w_vocab_size
    w_vocab_size = len(normalizer.vocabulary.tokenIndices)
    global p_vocab_size
    p_vocab_size = len(normalizer.vocabulary.posIndices)
    # nb of sentence positions taken to build
    # input vector representing a parse configuration
    model = TransitionClassifier().to(device)
    optimizer = getOptimizer(model.parameters())
    loss_function = nn.NLLLoss()
    # losses of validation set for each epoch
    epochLosses, validLosses = [], []
    sys.stderr.write(reports.tabs + "Network %s\n" % str(model) + reports.doubleSep)
    sys.stderr.write(reports.tabs + "Optimizer %s\n" % str(optimizer) + reports.doubleSep)
    # sys.stderr.write(reports.tabs + "Loss function %s\n" % str(loss_function) + reports.doubleSep)
    # ------------ if validation asked -------------
    validSeuil = configuration["model"]["train"]["validationSplit"]
    pointer = int(len(corpus.trainingSents) * (1 - validSeuil))
    trainSents = corpus.trainingSents[:pointer]
    validSents = corpus.trainingSents[pointer:]
    # ------------ training ------------------------
    iterationIdx = 0
    for epoch in range(trainConf["epochs"]):
        sys.stderr.write(reports.tabs + "Epoch %d....\n" % epoch)
        # epoch_loss = torch.Tensor([0])
        epochLoss = 0
        # shuffle sentences
        sentRanks = range(len(trainSents))
        shuffle(sentRanks)
        # cf. some of the dgs may not fulfill formal requirements
        #     for a gold transition sequence to be found (eg projectivity)
        usedSents = 0
        for i in sentRanks:
            sent = trainSents[i]
            # sys.stdout.write(str(sent) + '\n')
            # sys.stderr.write("\tLearning using sentence %d %s\n" % (i, gold_dg.sentid))
            iterationIdx += 1
            # reinitialize gradients for the whole sentence
            model.zero_grad()
            # and the lstm hidden states
            model.lstm_hidden_and_cell = init_lstm_hidden_and_cell()

            sentLoss = getSentLoss(sent, normalizer, model, loss_function)  # , device)
            # if a gold transition sequence was extracted all right,
            # backpropagate the full sentence loss
            if sentLoss:
                usedSents += 1
                epochLoss += sentLoss.item()
                # backpropagate
                sentLoss.backward()
                # update parameters
                optimizer.step()

        epochLosses.append(epochLoss)
        if not epoch:
            sys.stderr.write("Number of used sentences in train = %d\n" % usedSents)
        sys.stderr.write("Total loss for epoch %d: %f\n" % (epoch, epochLoss))
        # logstream.write("Total loss for epoch %d: %f\n" % (epoch, epoch_loss))
        # check validation losses
        if validSents:
            valLoss = getCorpusLoss(validSents, normalizer, model, loss_function)  # , device)
            sys.stderr.write("validation loss after epoch %d : %f\n" % (epoch, valLoss))
            validLosses.append(valLoss)
            # logstream.write("validation loss after epoch %d : %f\n" % (epoch, valLoss))
            # save model if validation loss has decreased Todo
            # if not epoch or valLoss <= validation_losses[-1]:
            # model.current_nb_epochs = epoch + 1
            # torch.save(model, opt['model_file'])

            # early stopping TODO réintégrer
            # elif trainConf["earlyStop"] and epoch and (l > validation_losses[-1]):
            #     sys.stderr.write("validation loss has increased (), stop and retrain using %d epochs\n" % epoch)
            #     logstream.write("validation loss has increased (), stop and retrain using %d epochs\n" % epoch)
            #     if validSeuil:
            #         # retrain on whole train+validation, using epoch-1 epochs
            #         configuration["model"]["train"]["validationSplit"] = 0
            #
            #         opt['nb_epochs'] = epoch  # (cf. epoch est décalé de 1)
            #         return train_transition_classifier(depgraphs, indices,
            #               transition_parser, config_feats_description, opt, logstream)
        # if no validation set: save iff loss on training set decreases
        # else: TODO réintégrer
        #     if (epoch == 0) or (epoch_loss < epoch_losses[-2]):
        #         # torch.save(classifier.state_dict(), args.model_file)
        #         # another_model = TransitionClassifier(*args, **kwargs)
        #         # the_model.load_state_dict(torch.load(PATH))
        #         model.current_nb_epochs = epoch + 1
        #         torch.save(model, opt['model_file'])

    return model


def getOptimizer(params):
    trainConf = configuration["model"]["train"]
    sys.stdout.write('# Network optimizer = {0}, learning rate = {1}\n'.format(trainConf["optimizer"], trainConf["lr"]))
    lr = trainConf["lr"]
    if trainConf["optimizer"] == 'sgd':
        return optim.SGD(params, lr=lr)
    if trainConf["optimizer"] == 'adam':
        return optim.Adam(params, lr=lr)
    if trainConf["optimizer"] == 'rmsprop':
        return optim.RMSprop(params, lr=lr)
    if trainConf["optimizer"] == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    if trainConf["optimizer"] == 'adadelta':
        return optim.Adadelta(params, lr=lr)
    if trainConf["optimizer"] == 'adamax':
        return optim.Adamax(params, lr=lr)  # lr = 0.002
    assert 'No optimizer found for training!'


def make_y(label):
    """
    cree un Tensor de shape (1,); i.e. on crée le label gold pour un mini-batch de 1
    """
    return torch.LongTensor([label])
