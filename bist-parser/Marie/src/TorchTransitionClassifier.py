#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author: marie candito

from __future__ import print_function
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from random import shuffle

from FeatureExtractor import *

torch.manual_seed(1)


# ##
# # marie:
# # 6 = length of sequence
# # 3 = batch size
# # 5 = input_size (at each position in sequence)
# input = torch.randn(6, 3, 5)
# # hidden and cell are of shape (num_layers * num_directions, batch, hidden_size)
# h0 = torch.randn(1,3,3)
# c0 = torch.randn(1,3,3)
# out, hidden = lstm(input, (h0,c0))
# print(out)
# print(hidden)


# voir pytorch_tutorial_nlp_dynamic_LSTM_CRF.ipynb


class TransitionClassifierLocal(nn.Module):

    """ version simple: le x d'une entrée est la concat des id des focused elements (S0, S1, B0, B1) 
    """
    # voir https://pytorch.org/docs/stable/nn.html#embedding pour la methode from_pretrained
    
    def __init__(self, indices, nb_labels, embedding_dim, nb_focus_elements):
        # nb_labels is the number of possible transitions
        # nb_focus_elements is the number of lexical nodes used as features
        super(TransitionClassifierLocal, self).__init__()

        vocab_size = indices.get_vocab_size('w')
        print("VOCAB_SIZE %d" % vocab_size)
        print("NB_FOCUS_ELEMENTS %d" % nb_focus_elements)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(nb_focus_elements * embedding_dim, 128)
        self.linear2 = nn.Linear(128, nb_labels)

    # from https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#word-embeddings-in-pytorch
    def forward(self, inputs):
        #inputs shape is (nb_focus_elements,)
        #print("INPUTS.SHAPE", inputs.shape)
        # self.embeddings(inputs) has shape (nb_focus_elements, embedding_dim)
        # .view(1,-1) concatenates the embeddings of all focus elements
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        scores = F.log_softmax(out, dim=1)
        return scores

    def predict_one(self, x):
        y_pred_vect = self(x)
        pred_score, y_pred = torch.max(y_pred_vect,1)
        return (y_pred, pred_score)

    def predict(self,x):
        """ return sorted predictions """
        y_pred_vect = self(x)
        #print(y_pred_vect)
        # A CREUSER, ici pq on obtient par ex: tensor([[-1.3893, -1.6119, -0.5956]])
        y_pred_vect = y_pred_vect[0]
        (sorted, indices) = torch.sort(y_pred_vect, descending=True)
        #sorted = [ a.item() for a in sorted ]
        #indices = [ a.item() for a in indices ]
        return (sorted, indices)

def select_rows(tensor, row_ids):
    """ extract given rows (= axis 0) from a given 2 dim tensor """
    tuple = [ tensor[i,:].view(1,-1) for i in row_ids ]
    return torch.cat(tuple)
    
class TransitionClassifier(nn.Module):

    """ version avec embeddings / lstm sur tous les mots de la phrase, 
    calcul de perte sur la phrase entière (somme sur les pertes à chaque transition)
    et update de gradient pour la phrase entière
    """
    # voir https://pytorch.org/docs/stable/nn.html#embedding pour la methode from_pretrained
    
    def __init__(self, indices, nb_labels, w_embedding_dim, p_embedding_dim, gru_or_lstm, rnn_hidden_dim, nb_focus_elements, use_pretrained_w_emb=False):
        """ if use_pretrained_w_emb, the indices.w_embeddings_matrix will be used """
        # nb_labels is the number of possible transitions (more precisely (transname/dep_label) pairs)
        # nb_focus_elements is the number of lexical nodes used as features
        super(TransitionClassifier, self).__init__()

        # strings <-> ids correspondences
        self.indices = indices

        # current nb of epochs used for training
        self.current_nb_epochs = None
        
        w_vocab_size = indices.get_vocab_size('w')
        p_vocab_size = indices.get_vocab_size('p')

        self.p_embeddings = nn.Embedding(p_vocab_size, p_embedding_dim)

        if not use_pretrained_w_emb:
            self.w_embeddings = nn.Embedding(w_vocab_size, w_embedding_dim)
        else:
            pretrained_w_emb_matrix = indices.w_embeddings_matrix
            
            if w_embedding_dim != pretrained_w_emb_matrix.shape[1]:
                sys.stderr.write("Error, pretrained embeddings are of size %d whereas %d is expected"
                                 % (pretrained_w_emb_matrix.shape[1], w_embedding_dim))
            if w_vocab_size != pretrained_w_emb_matrix.shape[0]:
                sys.stderr.write("Error, pretrained embeddings have a %d vocab size while indices have %d"
                                 % (pretrained_w_emb_matrix.shape[0], w_vocab_size))
            # here: use freeze=True to freeze pretrained embeddings
            self.w_embeddings = nn.Embedding.from_pretrained(pretrained_w_emb_matrix, freeze = False)
            
        self.embedding_dim = w_embedding_dim + p_embedding_dim
        
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = 2 #cf. kipgol16
        self.gru_or_lstm = gru_or_lstm
        
        # see https://pytorch.org/docs/stable/nn.html#lstm
        if gru_or_lstm == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim,
                               rnn_hidden_dim,
                               bidirectional = True,
                               num_layers = self.num_rnn_layers,
                               dropout = 0.5) # after internal lstm layer
        else:
            self.rnn = nn.GRU(self.embedding_dim,
                              rnn_hidden_dim,
                              bidirectional = True,
                              num_layers = self.num_rnn_layers,
                              dropout = 0.5) # after internal gru layer
        
        # * 2 because bidirectional
        self.linear1 = nn.Linear(nb_focus_elements * rnn_hidden_dim * 2, 128)
        
        # dropout here is very detrimental
        #self.dropout1 = nn.Dropout(p=0.5)
        
        self.linear2 = nn.Linear(128, nb_labels)

    def init_lstm_hidden_and_cell(self):
        """ init : h0 and c0 """
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # h_0 is of shape (num_layers * num_directions, batch, hidden_size):
        #                 tensor containing the initial hidden state for each element in the batch
        # same for c_0 = initial cell state
        # here num_directions = 2 (bidirectional), batch = 1
        return (torch.zeros(self.num_rnn_layers * 2, 1, self.rnn_hidden_dim),
                torch.zeros(self.num_rnn_layers * 2, 1, self.rnn_hidden_dim))

    def init_gru_hidden(self):
        """ init : h0 """
        # h_0 is of shape (num_layers * num_directions, batch, hidden_size):
        #                 tensor containing the initial hidden state for each element in the batch
        # here num_directions = 2 (bidirectional), batch = 1
        return torch.zeros(self.num_rnn_layers * 2, 1, self.rnn_hidden_dim)


    def compute_contextualized_embeddings(self, sent_w_ids, sent_p_ids):
        """ 
        input = sequence of word ids / sequence of pos ids of the sentence
        as output by get_sequence_feats_for_whole_sentence)

        (each are a LongTensor( list of ids )
        and contain as first element the not_existing_node, and the ROOT as second element,
        then all positions of the sentence

        output = tensor of shape (sequence length, self.embedding_dim)
        with rnn-contextualized vectors for each position, computed over non-contextualized embeddings
        """

        sent_length = len(sent_p_ids)
        
        sentence_w_embeddings = self.w_embeddings(sent_w_ids) # shape = len of sent (+2) , w_embedding_dim
        sentence_p_embeddings = self.p_embeddings(sent_p_ids) # shape = len of sent (+2) , p_embedding_dim

        # embedding for one word is concatenation of the word form embedding and the pos embedding
        sentence_embeddings = torch.cat( [ sentence_w_embeddings, sentence_p_embeddings ], 1 )

        # re-set to 0 the hidden vectors (TODO: other initialization?)
        if self.gru_or_lstm == 'lstm':
            # init hidden and cell states h0 c0
            self.rnn_hidden = self.init_lstm_hidden_and_cell()
        else:
            # init hidden state h0
            self.rnn_hidden = self.init_gru_hidden()
            
        #print("SENTENCE_EMBEDDINGS.SHAPE", sentence_embeddings.shape)
        #print("INPUT SHAPE TO RNN:", sentence_embeddings.view(len(sent_length, 1, -1).shape)
        #print("RNN_HIDDEN.shape:", self.rnn_hidden_and_cell[0].shape)

        # rnn takes the batch size as second dimension (here batch size=1)
        # rnn_hidden_seq = the output(hidden) vectors at each position in sequence
        # rnn_hidden = if lstm: the output and cell vectors at the last position
        #              if gru:  the output vector at the last position
        rnn_hidden_seq, self.rnn_hidden = self.rnn(sentence_embeddings.view(sent_length, 1, -1),
                                                   self.rnn_hidden)
        #print("LSTM_OUT.shape", lstm_out.shape)
        #print("LSTM_OUT output shape", lstm_out.view(len(sent_length), -1).shape)

        return rnn_hidden_seq.view(sent_length, -1)
       
        
    # from https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#word-embeddings-in-pytorch
    def forward(self, embeddings_of_sentence, focused_elements_lidxs):
        """ propagation avant, pour prediction d'une transition
        input = 
        - embeddings as output from compute_contextualized_embeddings
        (contextualized with rnn layer)

        - the list of the lidxs of the nodes to focus on, according to current configuration
        """

        # select the contextualized embeddings for the focused_elements (given configuration)
        focus_embeds = select_rows(embeddings_of_sentence, focused_elements_lidxs).view((1,-1))
        
        # print("EMBEDS.SHAPE", focus_embeds.shape)

        out = F.relu(self.linear1(focus_embeds))
        out = self.linear2(out)
        #out = self.dropout1(out) # detrimental
        # ne pas faire de softmax si fait dans la loss (CrossEntropyLoss au lieu de NLLLoss)
        scores = F.log_softmax(out, dim=1)
        return scores
        

    def predict(self, x, focused_elements_lidxs):
        """ prediction of score vector (for each transition)
        Returns sorted score list / corresponding class id list """

        # cet appel comprend l'appel de self.forward
        y_pred_vect = self(x, focused_elements_lidxs)

        # on obtient par ex, si 3 transitions: tensor([[-1.3893, -1.6119, -0.5956]])
        # (cf. minibatch de 1)
        y_pred_vect = y_pred_vect[0]
        (sorted_scores, sorted_indices) = torch.sort(y_pred_vect, descending=True)

        return (sorted_scores, sorted_indices)


def make_y(label):
    # cree un Tensor de shape (1,)
    # i.e. on crée le label gold pour un mini-batch de 1
    return torch.LongTensor([label])

# OBSOLETE
# def train_local_transition_classifier(gold_depgraphs, indices, transition_parser, embedding_dim, nb_focus_elements, opt):
#     """ version où on met à jour les paramètres à chaque transition (et pas pour une phrase complète) """
#     nb_sentences = len(gold_depgraphs)
#     nb_transitions = len(transition_parser.transitions.keys())

#     classifier = TransitionClassifierLocal(indices, nb_transitions, embedding_dim, nb_focus_elements)
# #    optimizer = optim.SGD(classifier.parameters(), lr=opt['lr'])
#     optimizer = optim.__dict__[opt['optimizer']](classifier.parameters(), lr=opt['lr'])
#     # rem: kipgol16 utilisent une multiclass margin loss, à peu près équivalente à torch.nn.MultiMarginLoss, à tester? (mais sans appliquer le softmax du coup)
#     loss_function = nn.NLLLoss()
#     epoch_losses = []

#     for epoch in list(range(opt['nb_epochs'])):

#         total_loss = torch.Tensor([0])
        
#         # shuffle sentences
#         sentence_ranks = list(range(nb_sentences))
#         shuffle(sentence_ranks)

#         for i in sentence_ranks:
#             gold_dg = gold_depgraphs[i]
#             print("Learning using sentence %d %s" % (i, gold_dg.sentid))

#             for (c,t,l) in transition_parser.yield_and_apply_next_static_gold_config(gold_dg):
#                 # if a gold transition sequence was extracted all right:
#                 if c:
#                     # id of the gold label (gold transition = pair (transition t,label l))
#                     y = indices.i_from_c( (t,l) )
#                     #@@print("gold trans %s is to apply on config: %s" % (t, c))

#                     x = torch.LongTensor( config_to_feat_vect(c, indices, gold_dg) )
#                     #print(x)
#                     classifier.zero_grad()

#                     # prediction
#                     y_pred_vect = classifier(x)

#                     # loss
#                     loss = loss_function(y_pred_vect, make_y(y))

#                     loss.backward()
#                     optimizer.step()

#                     # le .data ne rajoute que le tenseur (de taille 1 ici)
#                     total_loss += loss.data
#         epoch_losses.append(total_loss)
#     print("EPOCH_LOSSES %s" % str(epoch_losses))  # The loss decreased every iteration over the training data!
#     return classifier

# hinge multiclass loss, but adapted to the fact that several trans can be correct
# (we don't want to penalize high score for non-best but still correct transitions)
# cf. section 4.1 Kipgol16)
# DOES NOT WORK
def multiclass_hinge_loss(y_pred_vect, best_correct_t_id, best_incorrect_t_id):
    """ y_pred_vect : 1D Tensor containing the scores for each transition id """
#    sys.stderr.write(str(y_pred_vect))
#    sys.stderr.write(str(y_pred_vect.requires_grad))
#    sys.stderr.write(str(y_pred_vect.grad_fn))

    local_loss = max(0, 1 - y_pred_vect[best_correct_t_id] + y_pred_vect[best_incorrect_t_id])
#    sys.stderr.write(str(local_loss.requires_grad))

#    # recreate batch of size 1
#    loss = torch.LongTensor([ local_loss ])
#    sys.stderr.write(str(loss.requires_grad))
#    exit()
#    return loss
    return local_loss

def compute_corpus_loss(gold_dgs, indices, classifier, transition_parser, opt, logstream):
    """ returns the loss for a list of gold dgs
    return type = float
    """
    loss = 0.0
    for dg in gold_dgs:
#        l = compute_sentence_loss_with_static_oracle(dg, indices, classifier, transition_parser, opt, logstream)
        l = compute_sentence_loss_with_dynamic_oracle(dg, indices, classifier, transition_parser, opt, logstream)
        if l:
            loss += l.item() # transform into float
    return loss

def compute_sentence_loss_with_static_oracle(gold_dg, indices, classifier, transition_parser, opt, logstream):
    """ 
    returns the whole loss for the sentence in gold_dg  
    (return type is that of Torch.nn losses: differentiable Tensor of size 1)

    using a STATIC oracle for the transition system of transition_parser
    (see yield_and_apply_next_static_gold_config)

    unless no gold transition sequence could be found 
    (e.g. if gold_dg is a non proj tree and the transition set does not handle it)
    in which case returns None
    """
    # list of losses after each decision (transition) for the sentence
    sentence_losses = []

    (w_ids, p_ids, l_ids) = get_sequence_feats_for_whole_sentence(gold_dg, indices)
    # don't use lemma ids for now
            
    w_ids = torch.LongTensor( w_ids )
    p_ids = torch.LongTensor( p_ids )

    embeddings_of_sentence = classifier.compute_contextualized_embeddings( w_ids, p_ids )
    #logstream.write("sentence %s\n" % str(gold_dg.sentid))
    
    for (c,t,l) in transition_parser.yield_and_apply_next_static_gold_config(gold_dg):
        # if a gold transition sequence could not be found:
        # ((None,None) is received)
        if not c:
            return None
        #logstream.write("transition %s label %s\n" % (t, str(l)))
        # id of the gold label (gold transition = pair (transition t,label l))
        y = indices.i_from_c( (t,l) )
        #print("gold trans %s is to apply on config: %s" % (t, c))
        
        focused_lidxs = get_focused_elements(c, feats_description=FEATS1)
        # prediction
        y_pred_vect = classifier(embeddings_of_sentence, focused_lidxs)
        # loss
        loss = opt['loss_function'](y_pred_vect, make_y(y))

        # # multiclass_hinge_loss get best incorrect transitions
        # (sorted_scores, sorted_transition_ids) = torch.sort(y_pred_vect[0], descending=True)
        # if sorted_transition_ids[0] != y:
        #     best_incorrect_t_id = sorted_transition_ids[0]
        # else:
        #     best_incorrect_t_id = sorted_transition_ids[1]
        # loss = multiclass_hinge_loss(y_pred_vect[0], y, best_incorrect_t_id)
        
        sentence_losses.append(loss)
        
    # loss for whole sentence is simply sum of losses
    return sum(sentence_losses)


def compute_sentence_loss_with_dynamic_oracle(gold_dg, indices, classifier, transition_parser, opt, logstream):
    """ 
    returns the whole loss for the sentence in gold_dg  
    (return type is that of Torch.nn losses: differentiable Tensor of size 1)

    with a DYNAMIC oracle (non deterministic and complete) (Goldberg and Nivre TALC 2013)

    unless gold_dg is a non proj tree
    in which case returns None
    """
    # list of losses after each decision (transition) for the sentence
    sentence_losses = []

    (w_ids, p_ids, l_ids) = get_sequence_feats_for_whole_sentence(gold_dg, indices)
    # don't use lemma ids for now
            
    w_ids = torch.LongTensor( w_ids )
    p_ids = torch.LongTensor( p_ids )

    embeddings_of_sentence = classifier.compute_contextualized_embeddings( w_ids, p_ids )
    #logstream.write("sentence %s\n" % str(gold_dg.sentid))        

    lidxs = gold_dg.sorted_dependent_lidx()
    sentence = [gold_dg.get_lexnode(x).form for x in lidxs]

    c = transition_parser.get_initial_config(sentence)
    
    while (not transition_parser.is_config_terminal(c)):
        # we first need all the correct transitions
        (legal_unlabeled_trans, correct_labeled_trans) = transition_parser.get_legal_and_correct_transitions(c, gold_dg)
        #sys.stderr.write("LEGAL_UNLABELED_TRANS %s\n" % str(legal_unlabeled_trans))
        #sys.stderr.write("CORRECT_LABELED_TRANS %s\n" % str(correct_labeled_trans))
        if (legal_unlabeled_trans == None and correct_labeled_trans == None):
            #(special return of get_legal_and_correct_transitions
            # when gold_dg is non projective)
            return None
        
        # prediction 
        focused_lidxs = get_focused_elements(c, feats_description=FEATS1)
        y_pred_vect = classifier(embeddings_of_sentence, focused_lidxs)

        # compute best_correct transition, and best_incorrect transition

        # sort the transitions according to their score
        #            (batch of size 1, hence we take the first row: [0])
        (sorted_scores, sorted_transition_ids) = torch.sort(y_pred_vect[0], descending=True)

        highest_legal_t_id = None # will equal either highest_correct or highest_incorrect
        highest_correct_t_id = None
        highest_incorrect_t_id = None # needed for aggressive error exploration

        for t_id in sorted_transition_ids:
            t_id = t_id.item()
            # get the unlabeled transition name + the dep label associated to this transition id
            (t,l) = indices.i2c[t_id]

            if t in legal_unlabeled_trans:
                if highest_legal_t_id == None:
                    highest_legal_t_id = t_id
                if (t,l) in correct_labeled_trans or (t, "*ANYLABEL*") in correct_labeled_trans:
                    if highest_correct_t_id == None:
                        highest_correct_t_id = t_id
                elif highest_incorrect_t_id == None:
                        highest_incorrect_t_id = t_id

            # stop when everything was found
            if (highest_correct_t_id != None) and (highest_incorrect_t_id != None):        
                break

        if highest_correct_t_id == None:
            sys.stderr.write("WARNING: no correct trans\n")
            logstream.write("WARNING: no correct trans\n")

        # DOES NOT WORK
        # hinge multiclass loss, but adapted to the fact that several trans can be correct
        # (we don't want to penalize high score for non-best but still correct transitions)
        # cf. section 4.1 Kipgol16)
        # does not work, TODO: investigate how to create custom loss
        #loss = multiclass_hinge_loss(y_pred_vect[0], highest_correct_t_id, highest_incorrect_t_id)

        # compute loss (and thus afterwards backprop)
        # using as "gold" the highest_correct_t_id
        loss = opt['loss_function'](y_pred_vect, make_y(highest_correct_t_id))

        sentence_losses.append(loss)

        # ERROR EXPLORATION: which transition to apply
        # (independently of which transition to use for loss/backprop)

        # if prediction was correct
        if highest_correct_t_id == highest_legal_t_id:
            # aggressive error exploration
            # follow incorrect trans even if prediction was correct
            # TODO: should be when the incorrect is not too bad ...
            if highest_incorrect_t_id != None and np.random.random() < opt['aggressive_ee_prob']:
                t_id_to_apply = highest_incorrect_t_id
            else:
                t_id_to_apply = highest_correct_t_id
        else:
            if (opt['epoch'] >= opt['first_epoch_for_error_exploration']
                and np.random.random() < opt['ee_prob']):#):
                t_id_to_apply = highest_legal_t_id
                #sys.stderr.write("FOLLOW ERROR EXPL\n")
            else:
                #sys.stderr.write("FOLLOW CORRECT TRANS\n")
                t_id_to_apply = highest_correct_t_id
            
        # application of chosen transition
        (t,l) = indices.i2c[t_id_to_apply]
        transition_parser.apply_transition(c, t, l)
        #sys.stderr.write("transition %s label %s\n" % (t, str(l)))
                            
    # loss for whole sentence is simply the sum of losses
    return sum(sentence_losses)


def train_transition_classifier(depgraphs, nb_dep_labels, indices, transition_parser, config_feats_description, opt, logstream):
    """ version avec bi-RNN sur toute la phrase, et mise à jour des paramètres à chaque phrase (calcul de la perte pour une phrase complete) """
    nb_transitions = 0
    for x in transition_parser.transitions.values():
        if x.is_labeled:
            nb_transitions += nb_dep_labels
        else:
            nb_transitions += 1

    w_embedding_dim = opt['w_embedding_dim']
    p_embedding_dim = opt['p_embedding_dim']
    gru_or_lstm = opt['gru_or_lstm']
    rnn_hidden_dim = opt['rnn_hidden_dim']
    
    # nb of sentence positions taken to build
    # input vector representing a parse configuration
    nb_focus_elements = len(config_feats_description)
                            
    classifier = TransitionClassifier(indices,
                                      nb_transitions,
                                      w_embedding_dim,
                                      p_embedding_dim,
                                      gru_or_lstm,
                                      rnn_hidden_dim,
                                      nb_focus_elements)
#    optimizer = optim.SGD(classifier.parameters(), lr=opt['lr'])
    optimizer = optim.__dict__[opt['optimizer']](classifier.parameters(), lr=opt['lr'])
    # loss_function = nn.CrossEntropyLoss()
    # non cf. le log_softmax est fait dans le réseau
    opt['loss_function'] = nn.NLLLoss()
    epoch_losses = []
    # losses of validation set for each epoch
    validation_losses = []
    min_validation_loss = None
    
    for stream in [sys.stderr, logstream]:
        stream.write("Network %s\n" % str(classifier))
        stream.write("Optimizer %s\n" % str(optimizer))
        stream.write("Loss function %s\n" % str(opt['loss_function']))
    
    # ------- if validation set asked to be taken from the train set --------
    if opt['validation_split'] > 0:
        # take one dg out of 10 as validation set
        (val_dgs, train_dgs) = utils.split_list(depgraphs,
                                          proportion=opt['validation_split'],
                                          shuffle=False)
        sys.stderr.write("Train and validation set sizes %d / %d\n" % (len(train_dgs), len(val_dgs)))
        logstream.write("Train and validation set sizes %d / %d\n" % (len(train_dgs), len(val_dgs)))
    # ------- otherwise if a validation set is provided ---------------------   
    else:
        train_dgs = depgraphs
        if opt['validation_dgs']:
            val_dgs = opt['validation_dgs']
        else:
            val_dgs = None

    # ------------ training ------------------------
    iter = 0
    for epoch in list(range(1, opt['nb_epochs']+1)):

        sys.stderr.write("Epoch %d....\n" % epoch)
        opt['epoch'] = epoch
        
        #epoch_loss = torch.Tensor([0])
        epoch_loss = 0
        
        # shuffle sentences
        sentence_ranks = list(range(len(train_dgs)))
        ###shuffle(sentence_ranks)

        # cf. some of the dgs may not fulfill formal requirements 
        #     for a gold transition sequence to be found (eg projectivity)
        used_sentences = 0 

        for i in sentence_ranks:
            gold_dg = train_dgs[i]
            #sys.stderr.write("\tLearning using sentence %d %s\n" % (i, gold_dg.sentid))
            iter += 1

            # reinitialize gradients for the whole sentence
            classifier.zero_grad()

            #XXXX TODO:
            #TODO: debug:
            #la version compute_sentence_loss_with_dynamic_oracle avec p=0 est sensée etre equivalente a l\'oracle statique, or resultats chutent
            #A INVESTIGUER (tracer la sequence de transition utilisee?)
            #TODO: implementer de pouvoir repasser à une version non labelee

#            sentence_loss = compute_sentence_loss_with_static_oracle(gold_dg, indices, classifier, transition_parser, opt, logstream)
            sentence_loss = compute_sentence_loss_with_dynamic_oracle(gold_dg, indices, classifier, transition_parser, opt, logstream)
            # if a gold transition sequence was extracted all right,
            # backpropagate the full sentence loss
            if sentence_loss:
                used_sentences += 1
                epoch_loss += sentence_loss.item()
                # backpropagate
                sentence_loss.backward()
                # update parameters
                optimizer.step()

        epoch_losses.append(epoch_loss)
        if epoch == 1:
            sys.stderr.write("Number of used sentences in train = %d\n" % used_sentences)

        sys.stderr.write("Total loss for epoch %d: %f\n" % (epoch, epoch_loss))
        logstream.write("Total loss for epoch %d: %f\n" % (epoch, epoch_loss))

        # check validation losses
        if val_dgs:
            l = compute_corpus_loss(val_dgs, indices, classifier, transition_parser, opt, logstream)
            sys.stderr.write("validation loss after epoch %d : %f\n" % (epoch, l))
            logstream.write("validation loss after epoch %d : %f\n" % (epoch, l))

            if not(min_validation_loss) or l < min_validation_loss:
                min_validation_loss = l
                
            # save model if validation loss has decreased
            if epoch == 1 or (l <= min_validation_loss):
                classifier.current_nb_epochs = epoch 
                sys.stderr.write("validation loss has decreased, saving model, current_nb_epochs = %d\n" % epoch)
                logstream.write("validation loss has decreased, saving model, current_nb_epochs = %d\n" % epoch)
                torch.save(classifier, opt['model_file'])
                
            # early stopping
            elif opt['early_stopping'] and (epoch > 1) and (l > validation_losses[-1]):
                sys.stderr.write("validation loss has increased\n")
                logstream.write("validation loss has increased\n")
                # if the validation set was taken from the train (instead of provided externally)
                if opt['validation_split'] > 0:
                    # retrain on whole train+validation, using epoch-1 epochs
                    opt['validation_split'] = 0
                    opt['nb_epochs'] = epoch - 1 
                    return train_transition_classifier(depgraphs, indices, transition_parser, config_feats_description, opt, logstream)

            validation_losses.append( l )
        # if no validation set: save iff loss on training set decreases
        else:
            if (epoch == 1) or (epoch_loss < epoch_losses[-2]):
                #torch.save(classifier.state_dict(), args.model_file)
                #another_model = TransitionClassifier(*args, **kwargs)
                #the_model.load_state_dict(torch.load(PATH))
                classifier.current_nb_epochs = epoch
                sys.stderr.write("saving model, current_nb_epochs = %d\n" % epoch)
                logstream.write("saving model, current_nb_epochs = %d\n" % epoch)
                torch.save(classifier, opt['model_file'])
            
    return classifier

# TODO
#- reverifier le pourquoi de make_y, 1ere dim = mini-batch? mais quid des embeddings? j'envoie effectivement une entrée de la forme : tensor([  0,  21,  45,   0])

