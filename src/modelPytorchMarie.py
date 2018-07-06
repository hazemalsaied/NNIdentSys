#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


class TransitionClassifierLocal(nn.Module):
    """
        version simple: le x d'une entrée est la concat des id des focused elements (S0, S1, B0, B1)
    """

    # voir https://pytorch.org/docs/stable/nn.html#embedding pour la methode from_pretrained

    def __init__(self, indices, nb_labels, embedding_dim, nb_focus_elements):
        """
        :param indices:
        :param nb_labels: is the number of possible transitions
        :param embedding_dim:
        :param nb_focus_elements: is the number of lexical nodes used as features
        """
        super(TransitionClassifierLocal, self).__init__()
        vocab_size = indices.get_vocab_size('w')
        print("VOCAB_SIZE %d" % vocab_size)
        print("NB_FOCUS_ELEMENTS %d" % nb_focus_elements)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(nb_focus_elements * embedding_dim, 128)
        self.linear2 = nn.Linear(128, nb_labels)

    # from https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#word-embeddings-in-pytorch
    def forward(self, inputs):
        # inputs shape is (nb_focus_elements,)
        # print("INPUTS.SHAPE", inputs.shape)
        # self.embeddings(inputs) has shape (nb_focus_elements, embedding_dim)
        # .view(1,-1) concatenates the embeddings of all focus elements
        embeds = self.embeddings(inputs).view((1, -1))
        out = f.relu(self.linear1(embeds))
        out = self.linear2(out)
        scores = f.log_softmax(out, dim=1)
        return scores

    def predict_one(self, x):
        y_pred_vect = self(x)
        pred_score, y_pred = torch.max(y_pred_vect, 1)
        return y_pred, pred_score

    def predict(self, x):
        """ return sorted predictions """
        y_pred_vect = self(x)
        # print(y_pred_vect)
        # A CREUSER, ici pq on obtient par ex: tensor([[-1.3893, -1.6119, -0.5956]])
        y_pred_vect = y_pred_vect[0]
        sortedd, indices = torch.sort(y_pred_vect, descending=True)
        # sorted = [ a.item() for a in sorted ]
        # indices = [ a.item() for a in indices ]
        return sortedd, indices


def train_local_transition_classifier(gold_depgraphs, indices, transition_parser, embedding_dim, nb_focus_elements,
                                      opt):
    """ version où on met à jour les paramètres à chaque transition (et pas pour une phrase complète) """
    nb_sentences = len(gold_depgraphs)
    nb_transitions = len(transition_parser.transitions.keys())

    classifier = TransitionClassifierLocal(indices, nb_transitions, embedding_dim, nb_focus_elements)
    #    optimizer = optim.SGD(classifier.parameters(), lr=opt['lr'])
    optimizer = optim.__dict__[opt['optimizer']](classifier.parameters(), lr=opt['lr'])
    loss_function = nn.NLLLoss()
    epoch_losses = []

    for _ in list(range(opt['nb_epochs'])):

        total_loss = torch.Tensor([0])

        # shuffle sentences
        sentence_ranks = list(range(nb_sentences))
        shuffle(sentence_ranks)

        for i in sentence_ranks:
            gold_dg = gold_depgraphs[i]
            print("Learning using sentence %d %s" % (i, gold_dg.sentid))
            for (c, t) in transition_parser.yield_next_static_gold_config(gold_dg):
                # if a gold transition sequence was extracted all right:
                if c:
                    # id of the gold label (gold transition)
                    y = indices.i_from_c(t)
                    # @@print("gold trans %s is to apply on config: %s" % (t, c))
                    x = torch.LongTensor(config_to_feat_vect(c, indices, gold_dg))
                    # print(x)
                    classifier.zero_grad()
                    # prediction
                    y_pred_vec = classifier(x)
                    # loss
                    loss = loss_function(y_pred_vec, make_y(y))
                    loss.backward()
                    optimizer.step()
                    # le .data ne rajoute que le tenseur (de taille 1 ici)
                    total_loss += loss.data
        epoch_losses.append(total_loss)
    print("EPOCH_LOSSES %s" % str(epoch_losses))  # The loss decreased every iteration over the training data!
    return classifier


def make_y(label):
    """
    cree un Tensor de shape (1,); i.e. on crée le label gold pour un mini-batch de 1
    """
    return torch.LongTensor([label])
