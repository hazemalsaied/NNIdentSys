#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepconllreader import *
from GraphParser import *
from FeatureExtractor import *
from TorchTransitionClassifier import *
import argparse
import os



# TODO: faire un mode load pour test dgs, avec gestion mot inconnu
def load_dgs(input_stream, indices, mode='train'):
    
    dgs = []

    w2nbocc = defaultdict(int)

    # fake node for configurations in which some focused elements don't exist (e.g. B0)
    not_existing_node = LexicalNode('*N*', -2, cat='*N*')
    not_existing_node.set_feature('lemma', '*N*')

    # add to indices the indices for the fake node and the root node
    indices.add_lexnodes( [ not_existing_node, DUMMY_ROOT ] )

    # set of all labels encountered 
    labels = set([])

    for depparse in read_depparses_conll_stream(input_stream):
        if depparse != None:
        #sys.stdout.write(depparse.depgraph.to_string_deep_conll())
            dg = depparse.depgraph
            dgs.append(dg)
            labels = labels.union( [ x.label for x in dg.deps ] )
            # in train mode, we need to first count the nb of occs
            # to replace some occ by *UNK*
            if mode == 'train':
                update_w_counts(dg, w2nbocc)
    if mode == 'train':
        for dg in dgs:
            # some strings will be considered *UNK*
            # (here simply meaning they won't be registered in the indices)
            indices.add_lexnodes(dg.lexnodes.values(), unk_threshold=1, w2nbocc=w2nbocc)
    return (dgs, labels)


def update_w_counts(dg, w2nbocc):
    for node in dg.lexnodes.values():
        w2nbocc[node.form] += 1
        
# ---------- MAIN ------------------
if __name__ == "__main__":


    usage = """ various tests during implementation of transition-based graph parser """

    # read arguments
    argparser = argparse.ArgumentParser(usage = usage)
    argparser.add_argument('mode', choices=['train','test'], default='train')
    argparser.add_argument('conll_file', help='if mode is train, conll_file is used as training + validation set, otherwise it is used as test set', default=None)
    argparser.add_argument('model_file', help='if mode is train, trained model will be saved to that file, otherwise model will be loaded from it', default=None)
    argparser.add_argument('-l', '--log_file', help='if mode is train, log file with hyperparameters of model', default=None)
    argparser.add_argument('-p', '--pretrained_w_embeddings', help='pre-trained word embeddings file. NB: first line should contain nbwords embedding size', default=None)
    argparser.add_argument('-o', '--optimizer', help='The optimization algo (from pytorch optim module). Default=SGD', default='SGD')
    argparser.add_argument('-g', '--gru_or_lstm', help='gru or lstm for the recurrent NN, default=lstm', type=str, default='lstm')
    argparser.add_argument('-r', '--learning_rate', help='learning rate, default=0.1', type=float, default=0.1)
    argparser.add_argument('-n', '--nb_epochs', help='nb max d\'époques default=20', type=int, default=20)
    argparser.add_argument('-w', '--w_embedding_dim', help='size of word embeddings. Default=100', type=int, default=100)
    argparser.add_argument('-c', '--p_embedding_dim', help='size of POS embeddings. Default=12', type=int, default=12)
    argparser.add_argument('-s', '--validation_split', help='if set to value > 0 (and < 1), such fraction of the training set will be used as validation set', type=float, default=0)
    argparser.add_argument('-v', '--validation_conll_file', help='if provided, will be used to compute loss during training')
    argparser.add_argument('-e', '--early_stopping', action="store_true", help='if set and validation split >0 or validation_conll_file provided, training will stop as soon as validation loss increases. Note that IN ANY CASE, THE SAVED MODEL will be that with MINIMUM LOSS on validation set, early stopping is just to use if training should be stopped at the first loss increase. Default=False', default=False)
    argparser.add_argument('-x', '--ee_prob', help='proba for error exploration (if random < p, then explore error). Default 0 (=no exploration)', type=float, default=0.0)
    argparser.add_argument('-a', '--aggressive_ee_prob', help='proba for aggressive error exploration (if random < p, then explore error even if prediction is correct). Default 0 (=no exploration)', type=float, default=0.0)
    argparser.add_argument('-y', '--first_epoch_for_error_exploration', help='if error_exploration_p > 0, first epoch (epoch numbers starting at 1) at which error exploration is applied. Default=2', type=int, default=2)
    # TODO: implement deep case
    argparser.add_argument('-d', '--deep_and_surf', help='use surf dependency tree, deep dep graph or both structures (CURRENTLY only tested for surf). Default="surf"', default='surf')
    argparser.add_argument('-t', '--trace', action="store_true", help='print some traces. Default=False', default=False)
    args = argparser.parse_args()

    #parser = TransitionParser('ARCSTANDARD')
    parser = TransitionParser('ARCHYBRID')

    if args.mode == 'train':

        logstream = open(args.log_file, 'w')
        
        # before training: check whether we will be able to dump the model
        pdir = os.path.dirname(args.model_file)
        if not pdir: pdir = '.'
        # target is creatable if parent dir is writable
        if not os.access(pdir, os.W_OK):
            exit("Model file %s will not be writable!\n" % args.model_file)

        # non necessary choice: we take lstm_hidden_dim as sum of word and pos embeddings
        RNN_HIDDEN_DIM = args.w_embedding_dim + args.p_embedding_dim # as tuned in kipgol16

        opt = {
            'optimizer':args.optimizer,
            'lr':args.learning_rate,
            'nb_epochs':args.nb_epochs,
            'early_stopping':args.early_stopping,
            'w_embedding_dim':args.w_embedding_dim,
            'p_embedding_dim':args.p_embedding_dim,
            'gru_or_lstm':args.gru_or_lstm.lower(),
            'rnn_hidden_dim':RNN_HIDDEN_DIM,
            'validation_split':args.validation_split,
            'model_file':args.model_file,
            'first_epoch_for_error_exploration':args.first_epoch_for_error_exploration,
            'ee_prob':args.ee_prob,
            'aggressive_ee_prob':args.aggressive_ee_prob
        }

        if args.pretrained_w_embeddings:
            indices = Indices(w_embeddings_file=args.pretrained_w_embeddings)
        else:
            indices = Indices()

        instream = open(args.conll_file)
        (gold_dgs, dep_labels) = load_dgs(instream, indices, mode='train')

        # now that we have both the parser (with unlabeled transition names)
        # and the dependency labels
        # we add the ids for the labeled transitions
        indices.add_indices_of_labeled_transitions(parser, dep_labels)
        
        if args.validation_conll_file:
            valstream = open(args.validation_conll_file)
            opt['validation_split'] = 0
            (opt['validation_dgs'], dummy) = load_dgs(valstream, indices, mode='test')

        for stream in [sys.stderr, logstream]:
            stream.write("Training set = %s\n" % args.conll_file)
            stream.write("Pretrained embeddings = %s\n" % str(args.pretrained_w_embeddings))
            if args.validation_conll_file:
                stream.write("validation_conll_file = %s\n" % args.validation_conll_file)
            for o in ['w_embedding_dim', 'p_embedding_dim', 'gru_or_lstm', 'rnn_hidden_dim', 'validation_split', 'model_file', 'optimizer', 'lr', 'nb_epochs', 'ee_prob', 'aggressive_ee_prob', 'first_epoch_for_error_exploration']:
                stream.write("opt %s = %s\n" % (o, str(opt[o])))
            stream.write("VOCAB_SIZE %d\n" % indices.get_vocab_size('w'))
            stream.write("NB_FOCUS_ELEMENTS %d\n" % len(FEATS1))

        # build and train torch neural network
        # FEATS1 is defined in FeatureExtractor :
        # defines the elements of a configuration to take into account to build the input features of a configuration
        classifier = train_transition_classifier(gold_dgs, len(dep_labels), indices, parser, FEATS1, opt, logstream)

    # test mode
    else:
        sys.stderr.write("Parsing %s with %s ...\n" % (args.conll_file, args.model_file))
        classifier = torch.load(args.model_file)
        sys.stderr.write("loaded model %s, nb epochs=%d\n" % (args.model_file, classifier.current_nb_epochs))
        indices = classifier.indices
        instream = open(args.conll_file)
        i = 0
        for dp in read_depparses_conll_stream(instream):
            gold_dg = dp.depgraph
            if not i % 100:
                sys.stderr.write("Parsing sentence %d %s\r" % (i, dp.sentid))
            dg = parser.greedy_parse(gold_dg, classifier, indices)
            print(dg.to_string_conll())
            i += 1
            
        
