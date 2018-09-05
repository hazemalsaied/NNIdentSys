#!/usr/bin/env python
# -*- coding: utf-8 -*-

# marie candito

from GraphParser import *
from deepconllreader import *
import numpy as np

# principe gestion mots inconnus:
# en amont: gérer de manière précise quelle chaînes sont entrées ds le vocabulaire et ont un indice
#           et lesquelles sont unk
# ensuite, pour les chaînes absentes de Indices, on utilise l'indice de *UNK*

#===================================================
# Features
#===================================================

# True: also get feats from dependents of node
# (usually for stack nodes, and sometimes b0, depending on the transition system)
FEATS1 = [ ('B', 0, True),
           ('S', 0, True),
           ('S', 1, True),
           ('B', 1, False)
           ]

def get_sequence_feats_for_whole_sentence(gold_dg, indices):
    """ given a depgraph (could be a more simple structure)
    returns 3 lists of ids (one for word ids / pos ids / lemma ids)
    Each list has length = 2 + len of sentence
    - special first position reserved for the 'not_existing_node'
    - second position for the ROOT node
    - and then words of the sentence

    Is used at training and parsing time,
    but all strings have already been seen and registered in indices (via load_dgs)
    unk strings in indices are mapped to the idx of *UNK* for each vocab_type
    """
    triples = []
    for lidx in gold_dg.sorted_lidx():
        n = gold_dg.lexnodes[lidx]
        triples.append( (n.form, n.cat, n.get_feature_value('lemma')) )
    
    # add at beginning triple for not_existing_node, and triple for ROOT node
    # not_existing_node => rank 0
    # root => rank 1
    # lidx 0 => 2 etc...
    triples = [ ('*N*', '*N*', '*N*'),
                (DUMMY_ROOT.form, DUMMY_ROOT.cat, DUMMY_ROOT.get_feature_value('lemma')) ] + triples

    # transformation of strings to ids, for each type of vocab (forms, pos, lemmas)
    # NB: if string in unknown in index, the id of *UNK* is returned by i_from_s
    triples_ids =    [ (indices.i_from_s('w', t[0], create_new=False),
                        indices.i_from_s('p', t[1], create_new=False),
                        indices.i_from_s('l', t[2], create_new=False)) for t in triples ]
    
    return zip(*triples_ids)

# obsolete?
def add_feats_from_node(feats_dict, node, structured=False, use_lemma=True):
    """ extract feats from a LexicalNode
    If structured is True, features from dependents will also be extracted 
    """
    if node:
        feats_dict['w'].append(node.form)
        feats_dict['p'].append(node.cat)
        if use_lemma:
            feats_dict['l'].append(node.get_feature_value('lemma'))
    else:
        feats_dict['w'].append(None)
        feats_dict['p'].append(None)
        if use_lemma:
            feats_dict['l'].append(None)
    # TODO HERE: if structured => call add_feats_from_node on dependents, with structured=False

def get_focused_elements(c, feats_description=FEATS1):
    """
    returns the ordered list of "focused elements" in a configuration
    (list of lidxs)
    """
    lidxs = []
    for (s_or_b, rk, structured) in feats_description:
        s_or_b = c.__dict__[s_or_b]
        if len(s_or_b) > rk:
            lidx = s_or_b[rk]
            # à voir: pour l'instant la config ne contient pas les gold lexical nodes
            #node = c.dg.get_lexnode(lidx)
            #lidxs.append(lidx)
            # decalage de 2 (not_existing_node, dummy_root sont en premier...)
            lidxs.append(lidx + 2)
        else:
            #lidxs.append(None)
            # lidx of not_existing_node is -2
            lidxs.append(-2)
    return lidxs


def extract_feats_for_config(c, gold_dg, feats_description=FEATS1):
    """ 
    OBSOLETE
    returns a dictionary with vocab_type as keys 
          'w' for word form
          'p' for part-of-speech
          'l' for lemma
        For each vocab_type, the value is the ordered list of atomic features
        for each of the focused elements
    """
    #print("FEATS DESCRIPTION", feats_description)
    feats_dict = defaultdict(list)

    for lidx in get_focused_elements(c, feats_description):
        # todo here: check quid du DUMMYROOT versus "noeud inexistant" dans la config
        # faire une préextraction des traits une bonne fois pour toutes, au lieu de trimballer tout le dg ?
        # 
        if lidx:
            node = gold_dg.get_lexnode(lidx)
        else:
            node = None
        add_feats_from_node(feats_dict, node)
        
    return feats_dict

def transform_feat_strings_to_indices(feats_dict, indices, create_new=True):
    """ indices manages a mapping from atomic strings to indices, for a certain list of vocab_types
    For each vocab_type appearing in the feats_dict, this method transforms the atomic strings into their index
    """
    out_feats_dict = {}
    for (vocab_type, l) in feats_dict.items():
        out_feats_dict[vocab_type] = [ indices.i_from_s(vocab_type,x, create_new) for x in l ]
    return out_feats_dict

def config_to_feat_vect(c, indices, gold_dg):
    """ transform a configuration into vectorial representation """
    feats_dict = extract_feats_for_config(c, gold_dg)
    #@@print("with features: %s" % str(feats_dict))
    id_feats_dict = transform_feat_strings_to_indices(feats_dict, indices, create_new=False)
    #@@print("with idfeatures: %s" % str(id_feats_dict))
    # for now : use only the word forms
    #return torch.LongTensor(id_feats_dict['w'])
    return id_feats_dict['w']


class Indices:
    """ class to handle the correspondances from classes to indices and from words to indices correspondances """
    
    def __init__(self, vocab_types= ['w','p','l'], use_bias=True, w_embeddings_file=''):
        """ vocab_types are for the different types of "vocabulary" : word forms, pos, lemma... """
        # id to class
        self.i2c = []
        # class to id
        self.c2i = {}

        self.vocab_types = vocab_types
        # for each vocab type, initialize a dict from string to id, and a list
        self.i2s = {}
        self.s2i = {}
        for v in vocab_types:
            # add the *UNK*  special element for all vocab_types
            self.i2s[v] = [ '*UNK*' ]
            self.s2i[v] = { '*UNK*' : 0 }

        if w_embeddings_file:
            self.load_embeddings_from_scratch(w_embeddings_file)
        else:
            self.iw2embedding = None
            self.w_embedding_dim = 0


    def get_vocab_size(self, vocab_type):
        return len(self.i2s[vocab_type])

    def get_nb_classes(self):
        return len(self.i2c)
    
    def i_from_c(self, c):
        """ adds the class c if unknown yet, and retrieves index of class c 
        NB: in the case of labeled dependency parsing, the class can be a tuple (trans_name, dep label)
        """
        if c in self.c2i:
            return self.c2i[c]
        self.c2i[c] = len(self.i2c)
        #@print("CLASS %s associated to index %d" %(c, self.c2i[c]))
        self.i2c.append(c)
        return self.c2i[c]

    def i_from_s(self, vocab_type, w, create_new=True):
        """ returns index of w , in the vocabulary of type vocab_type
            if w is not already known,
             if create_new is False: 
                  returns the id of *UNK* ,
             otherwise:
                  creates a new index for w and returns it 
                  If pretrained word embeddings are used, 
                  creates a random embedding for w
        """
        if w in self.s2i[vocab_type]:
            return self.s2i[vocab_type][w]
        if not create_new:
            #return None
            return self.s2i[vocab_type]['*UNK*']
        self.s2i[vocab_type][w] = len(self.i2s[vocab_type])
        #@print("WORD %s associated to id %i" % (w, len(self.i2w)))
        self.i2s[vocab_type].append(w)
        if vocab_type == 'w' and self.iw2embedding:
            # for words unknown in embeddings:
            # random vector between -1 and 1  (b-a)*sample -a
            self.iw2embedding.append( 2*np.random.random(self.w_embedding_dim) - 1 )

        return self.s2i[vocab_type][w]

    def load_embeddings_from_scratch(self, embeddings_file):
        """
        Loads txt file containing embeddings 
        @param embeddings_file: vectors associated to words (or strings)
        First line contains : nb_words w_embedding_dim
        
        - fills self.iw2embedding list frow w id to embedding
        - sets self.w_embedding_dim
    
        """
        instream = open(embeddings_file)
        iw2embedding = []
        # reading nb_words and w_embedding_dim from first line
        line = instream.readline()
        line = line[:-1]
        (nb_words, w_embedding_dim) = [ int(x) for x in line.split(' ') ]
        self.w_embedding_dim = w_embedding_dim
        self.iw2embedding = []
        
        # if indices is not empty (eg *UNK*)
        i = self.get_vocab_size('w')
        while i > len(self.iw2embedding):
            # random vector between -1 and 1  (b-a)*sample -a
            self.iw2embedding.append( 2 * np.random.random(self.w_embedding_dim) - 1 )
            #self.iw2embedding.append( [ 0 for j in range(w_embedding_dim) ] )
        

        #self.embeddings_weights = np.zeros( (nb_words, w_embedding_dim) )
        line = instream.readline()
        while line:
            i += 1
            line = line[:-1].strip() # trailing space
            cols = line.split(" ")
            w = cols[0]
            vect = [float(x) for x in cols[1:]]
            self.s2i['w'][w] = i
            self.i2s['w'].append(w)
            self.iw2embedding.append(vect)
            
            line = instream.readline()


    def set_embeddings_matrix(self):
        """ construction of the torch.FloatTensor for embeddings
         (usable by the nn.Embedding.from_pretrained method)
        NB: to be done once all embeddings are loaded , plus random embeddings 
        for words in encountered in training data but unkwown in pre-trained embeddings 
        """
        n = self.get_vocab_size('w')
        self.w_embeddings_matrix = torch.FloatTensor( (n, self.w_embedding_dim) )
        for i in range(n):
            self.w_embeddings_matrix[i] = torch.FloatTensor( self.iw2embedding[i] )

    # obsolete
    def add_ids(self, vocab_type, strings):
        for s in strings:
            if s not in self.s2i[vocab_type]:
                self.s2i[vocab_type][s] = len(self.i2s[vocab_type])
                #@print("WORD %s associated to id %i" % (w, len(self.i2w)))
                self.i2s[vocab_type].append(s)

    def add_lexnodes(self, lexnodes, unk_threshold=0, w2nbocc=None ):
        """ add indices for the form / cat / lemma strings of the nodes,
        unless threshold is > 0:
        in such cases, only wforms having more than threshold occurrences 
        in w2nbocc are added an index,
        eventually the absence in index will cause replacement by special token
        (see ??? method, both in train and test)
        """
        
        l = [ (x.form, x.cat, x.get_feature_value('lemma')) for x in lexnodes ]
        for node in lexnodes:
            (w, p, l) = (node.form, node.cat, node.get_feature_value('lemma'))
            if not(unk_threshold) or w2nbocc[w] > unk_threshold:
                if 'w' in self.vocab_types:
                    self.i_from_s('w', w, create_new=True) 
                if 'p' in self.vocab_types:
                    self.i_from_s('p', p, create_new=True) 
                if 'l' in self.vocab_types:
                    self.i_from_s('l', l, create_new=True)
                    
    def add_indices_of_labeled_transitions(self, parser, dep_labels):
        """ adds the indices for the labeled transitions """
        for trans in parser.transitions.values():
            if trans.is_labeled:
                for label in dep_labels:
                    self.i_from_c( (trans.name, label) )
            else:
                # cf. label should be None for the transitions that do not add any arc
                self.i_from_c( (trans.name, None) )
    
