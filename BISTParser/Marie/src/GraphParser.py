#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author: marie candito

import sys
import argparse
from collections import defaultdict
#from LightDepGraph import *
from DepGraph import *
from copy import deepcopy
from TorchTransitionClassifier import *

# key = name of transition system
# val = list of transition names in this system
SYSTEM2TRANS = {}

# a transition system should define all the following methods / variables:
# - list of transition names in SYSTEM2TRANS
# - SYSTEMNAME_get_initial_config
# - SYSTEMNAME_is_config_terminal
# - SYSTEMNAME_get_static_gold_transitions
# - SYSTEMNAME_get_legal_and_correct_transitions (None if no dynamic oracle available)
# for each transition TRANS
#   - SYSTEMNAME_TRANS_is_legal
#   - SYSTEMNAME_TRANS_apply

# and if the transition system is compatible with a dynamic oracle:
# for each transition TRANS
#   - SYSTEMNAME_TRANS_unlab_cost : which returns the nb of gold arcs that become unreachable if the transition were applied to the configuration

# ------------------------------------------- #
# ARCSTANDARD TRANSITION SYSTEM
# ------------------------------------------- #

# True/False : whether the transition adds an arc, and hence needs a label to be applied
SYSTEM2TRANS['ARCSTANDARD'] = [('SHIFT',False), ('LEFTARC',True), ('RIGHTARC',True)]

ARCSTANDARD_get_legal_and_correct_transitions = None

def ARCSTANDARD_get_initial_config(sentence):
    """ return initial configuration for given sentence for ARCSTANDARD 
    @parameter: sentence (list of strings)
    @rtype=Config instance """
    return Config( list(range(len(sentence))) , [ DUMMY_ROOT_LIDX ] , sentence)

def ARCSTANDARD_is_config_terminal(c):
    """ terminal configuration for ARCSTANDARD (=buffer empty) 
     add also that the stack contains ROOT only, otherwise, some items may remain unattach """
    return len(c.B) == 0 and len(c.S) == 1

# Condition for applicability of transitions
# ------------------------------------------

def ARCSTANDARD_SHIFT_is_legal(c):
    """ buffer should have at least 2 elements (otherwise if stack is not empty, no transition will ever consume stack)
    unless b0 is the root (to move back to stack, which should be empty ...)"""
    #return len(c.B) > 0
    return len(c.B) > 1 or (c.B[0] == DUMMY_ROOT_LIDX)

def ARCSTANDARD_LEFTARC_is_legal(c):
    return len(c.B) and len(c.S) and c.S[0] > DUMMY_ROOT_LIDX

def ARCSTANDARD_RIGHTARC_is_legal(c):
    return len(c.B) and len(c.S)

# Application of transitions
# --------------------------
def ARCSTANDARD_SHIFT_apply(c, l):
    # mv B0 to S0
    c.S = [c.B[0]] + c.S
    del c.B[0]

def ARCSTANDARD_LEFTARC_apply(c, l):
    # add arc from B0 to S0 with label l; remove S0
    c.dg.add_dep_from_lidx(c.B[0], c.S[0], label=l)
    del c.S[0]

def ARCSTANDARD_RIGHTARC_apply(c, l):
    # arc from S0 to B0 with label l; remove B0 ; move S0 to B0  (hence replace B0 by S0 and remove S0)
    c.dg.add_dep_from_lidx(c.S[0], c.B[0], label=l)
    c.B = [ c.S[0] ] + c.B[1:]
    del c.S[0]


# Static gold transition sequence
# ------------------------
def ARCSTANDARD_get_static_gold_transitions(gold_dg, keep_configs=False):
    """ gets the transition sequence from a given (gold) depgraph 
    Implemented priority order: add arc whenever possible (note left/right are not possible at same time):
                     gold dep exist, and dependent has all its dependents already

    If gold_dg not reachable using ARCSTANDARD (i.e. tree not projective),
    returns (None, None, None)

    @return: (transitions, configs )
    @rtype: (list of tuples (str, str or None),  None or list of Config instances)
            configs is not None only if keep_configs is True
            ( transitions[i] is to apply on configs[i] , hence transitions has one less element than transitions

    NB: the transitions are pairs (transition name, dep label)

    ** dep label MUST BE SET to None for the transitions that do not add any arc **

    """
    lidxs = gold_dg.sorted_dependent_lidx()
    sentence = [gold_dg.get_lexnode(x).form for x in lidxs]

    c0 = ARCSTANDARD_get_initial_config(sentence)

    # list of configs, list of transitions 
    #configs = [ c0 ]
    others = []
    transitions = []
    # plus efficace de recalculer la config initiale
    #c = deepcopy(c0)
    c = ARCSTANDARD_get_initial_config(sentence)

    # for each lidx, precomputation of nb of gold dependencies having this lidx as head
    # (dico instead of list, so that DUMMY_ROOT_LIDX can be added without ambiguity
    nb_dep_to_add = dict( [ (x, len(gold_dg.get_dep_by_governor(x))) for x in lidxs + [DUMMY_ROOT_LIDX]] )

    while not(ARCSTANDARD_is_config_terminal(c)):

        l = None
        #print("NB OF DEPENDENTS %s" % str(nb_dep_to_add))
        if ARCSTANDARD_RIGHTARC_is_legal(c) and gold_dg.get_such_dependency(c.S[0], c.B[0]):
            # the dependent B0 must have all its dependents already
            if nb_dep_to_add[c.B[0]] == 0:
                t = 'RIGHTARC'
                nb_dep_to_add[c.S[0]] -= 1
                l = gold_dg.get_such_dependency(c.S[0], c.B[0]).label
            # if we are here, then left is not possible (input is not a digraph)
            elif ARCSTANDARD_SHIFT_is_legal(c):
                t = 'SHIFT'
            else:
                t = None
        # for LEFT, by construction, no need to check whether S0 has all dependents already
        elif ARCSTANDARD_LEFTARC_is_legal(c) and gold_dg.get_such_dependency(c.B[0], c.S[0]):
            t = 'LEFTARC'
            nb_dep_to_add[c.B[0]] -= 1
            l = gold_dg.get_such_dependency(c.B[0], c.S[0]).label
        elif ARCSTANDARD_SHIFT_is_legal(c):
            t = 'SHIFT'
        else:
            t = None

        if t==None:
            #sys.stderr.write("WARNING: no gold sequence found for tree\n")
            return (None, None, None)
            
        transitions.append( (t,l) )

        # application of transition to get new configuration:
        applyfuncname = 'ARCSTANDARD_'+t+'_'+'apply'
        if keep_configs:
            newc = deepcopy(c)
            eval(applyfuncname)(newc, l)
            others.append(newc)
            c = newc
        else:
            eval(applyfuncname)(c, l)
        
    return (transitions, c0, others)


# ------------------------------------------- #
# ARCHYBRID TRANSITION SYSTEM
# ------------------------------------------- #

# pairs (transname, is_labeled)
SYSTEM2TRANS['ARCHYBRID'] = [('SHIFT',False), ('LEFTARC',True), ('RIGHTARC',True)]


def ARCHYBRID_get_initial_config(sentence):
    """ return initial configuration for given sentence for ARCHYBRID 
    @parameter: sentence (list of strings)
    @rtype=Config instance """
    return Config( list(range(len(sentence))) , [ DUMMY_ROOT_LIDX ] , sentence)

def ARCHYBRID_is_config_terminal(c):
    """ terminal configuration for ARCHYBRID (=buffer empty, stack contains ROOT only) """
    return len(c.B) == 0 and len(c.S) == 1

# Condition for applicability of transitions
# ------------------------------------------

def ARCHYBRID_SHIFT_is_legal(c):
    """ buffer is not empty """
    return len(c.B) > 0

def ARCHYBRID_LEFTARC_is_legal(c):
    """ buffer and stack are not empty, and s0 is not ROOT """
    return len(c.B) and len(c.S) and c.S[0] > DUMMY_ROOT_LIDX

def ARCHYBRID_RIGHTARC_is_legal(c):
    # diff from ARCSTANDARD
    """ stack has at least two elements ( =/= ARCSTANDARD ) """
    return len(c.S) > 1

# Application of transitions
# --------------------------
def ARCHYBRID_SHIFT_apply(c, l):
    # (== ARCSTANDARD)
    # mv B0 to S0 
    c.S = [c.B[0]] + c.S
    del c.B[0]

def ARCHYBRID_LEFTARC_apply(c, l):
    # (== ARCSTANDARD)
    # add arc from B0 to S0 with label l; remove S0
    c.dg.add_dep_from_lidx(c.B[0], c.S[0], label=l)
    del c.S[0]

def ARCHYBRID_RIGHTARC_apply(c, l):
    # sole difference with ARCSTANDARD (note that S1 cannot receive any left deps)
    # arc from S1 to S0 with label l; remove S0 
    c.dg.add_dep_from_lidx(c.S[1], c.S[0], label=l)
    del c.S[0]

# Cost of transitions (cf. Goldberg and Nivre 2013)
# -------------------------------------------------
# see Goldberg and Nivre talc 2013 p.411

# cost of a transition = nb of arcs that BECOME IMPOSSIBLE TO REACH if the transition is applied
# NB: cost ≠ nb of wrong arcs added by the transition:
#     we don't need to check whether the added arc is in gold or not
#     If it is not, the corresponding gold arc will be counted as unreachable at this point or some other point

def ARCHYBRID_get_legal_and_correct_transitions(c, gold_dg):
    """ get the list of legat unlabeled trans
    and the list of correct labeled transitions : legal transitions with cost 0, that add the arc of the right label
    @return: (list of transname, list of (transname, deplabel) pairs )
    NB: for transitions for which labeling is irrelevant (SHIFT), the label is None 
    REM: in labeled mode, it is faster to precompute all the correct transitions rather than compute correctness for all labeled transitions """

    if gold_dg.is_projective() == False:
        return (None, None)
    correct_lab_trans = []
    legal_unlab_trans = []

    if ARCHYBRID_SHIFT_is_legal(c):
        legal_unlab_trans.append("SHIFT")
        if ARCHYBRID_SHIFT_unlab_cost(c, gold_dg) == 0:
            correct_lab_trans.append( ("SHIFT", None) )
            
    if ARCHYBRID_RIGHTARC_is_legal(c):
        legal_unlab_trans.append("RIGHTARC")
        if ARCHYBRID_RIGHTARC_unlab_cost(c, gold_dg) == 0:
            # by construction, if the cost is 0, then
            # either
            # - the arc added by this rightarc is gold
            #    and then the only correct labeled transition is the one with the gold label
            # - or the tree is non projective (already checked out)
            # - or the gold arc was already made unreachable by previous transitions
            #   (eg already attached to some wrong head)
            #   (and the cost of it has already been counted)
            dep = gold_dg.get_such_dependency(c.S[1], c.S[0])
            label = "*ANYLABEL*" if dep == None else dep.label
            correct_lab_trans.append( ("RIGHTARC", label) )
            
    if ARCHYBRID_LEFTARC_is_legal(c):
        legal_unlab_trans.append("LEFTARC")
        if ARCHYBRID_LEFTARC_unlab_cost(c, gold_dg) == 0:
            # idem
            dep = gold_dg.get_such_dependency(c.B[0], c.S[0])
            label = "*ANYLABEL*" if dep == None else dep.label
            correct_lab_trans.append( ("LEFTARC", label) )

    return (legal_unlab_trans, correct_lab_trans)
    
def ARCHYBRID_SHIFT_unlab_cost(c, gold_dg):
    """ cost of applying shift """
    # nb of arcs (B0, s) for any s in the stack
    cost = len([ s_lidx for s_lidx in c.S if gold_dg.get_such_dependency(c.B[0], s_lidx) ])
    # plus nb of arcs (s, B0) for any s in the stack except S0
    cost += len([ s_lidx for s_lidx in c.S[1:] if gold_dg.get_such_dependency(s_lidx, c.B[0]) ])
    return cost

def ARCHYBRID_LEFTARC_unlab_cost(c, gold_dg):
    """ cost of applying leftarc """

    # nb of arcs (Bi, S0) for any Bi in the buffer except B0
    # +1 for the arc (S1, S0) if it exists
    # nb of arcs (S0, Bi) for any Bi in the buffer
    
    headsofS0 = [ x.governor.lidx for x in gold_dg.get_dep_by_dependent(c.S[0]) ]
    depsofS0 = [ x.dependent.lidx for x in gold_dg.get_dep_by_governor(c.S[0]) ]

    bufferset = set(c.B[1:]) # buffer minus B0
    cost = len(bufferset.intersection(headsofS0)) + len(bufferset.intersection(depsofS0))
    if c.S[1] in headsofS0:
        cost += 1
    if c.B[0] in depsofS0:
        cost += 1
    return cost

def ARCHYBRID_RIGHTARC_unlab_cost(c, gold_dg):
    """ cost of applying rightarc """

    # nb of arcs (S0,b) or (b,S0) for any item b in the buffer
    headsofS0 = [ x.governor.lidx for x in gold_dg.get_dep_by_dependent(c.S[0]) ]
    depsofS0 = [ x.dependent.lidx for x in gold_dg.get_dep_by_governor(c.S[0]) ]

    bufferset = set(c.B)
    return len(bufferset.intersection(headsofS0)) + len(bufferset.intersection(depsofS0))

# Static gold transition sequence
# ------------------------

def ARCHYBRID_get_static_gold_transitions(gold_dg, keep_configs=False):
    """ gets the transition sequence from a given (gold) depgraph 
    Implemented priority order: add arc whenever possible (note left/right are not possible at same time):
                     gold dep exist, and dependent has all its dependents already

    If gold_dg not reachable using ARCHYBRID (i.e. tree not projective),
    returns (None, None, None)

       @return: (transitions, initial_config, subsequent_configs ] )
                The transitions list is a list of pairs (transname, dep label)
       @rtype: (list of tuples (str, str or None),  Config, None or list of Config instances)
            subsequent_configs is not None only if keep_configs is True
    if keep_config is True, the full seq of configs is c0 + subsequent_configs,
    in which transitions[i] is to apply on the ith config

    NB: the transitions are pairs (transition name, dep label)
    ** dep label MUST BE SET to None for the transitions that do not add any arc **

    """
    lidxs = gold_dg.sorted_dependent_lidx()
    sentence = [gold_dg.get_lexnode(x).form for x in lidxs]

    c0 = ARCHYBRID_get_initial_config(sentence)

    # list of configs, list of transitions 
    #configs = [ c0 ]
    others = []
    transitions = [  ]
    # plus efficace de simplement recalculer la config initiale
    #c = deepcopy(c0)
    c = ARCHYBRID_get_initial_config(sentence)
    
    # for each lidx, precomputation of nb of gold dependencies having this lidx as head
    # (dico instead of list, so that DUMMY_ROOT_LIDX can be added without ambiguity
    nb_dep_to_add = dict( [ (x, len(gold_dg.get_dep_by_governor(x))) for x in lidxs + [DUMMY_ROOT_LIDX]] )

    while not(ARCHYBRID_is_config_terminal(c)):

        l = None
        # the dependent S0 must have already received all its dependents
        if ARCHYBRID_RIGHTARC_is_legal(c) \
           and gold_dg.get_such_dependency(c.S[1], c.S[0]) \
           and nb_dep_to_add[c.S[0]] == 0:
            t = 'RIGHTARC'
            nb_dep_to_add[c.S[1]] -= 1
            l = gold_dg.get_such_dependency(c.S[1], c.S[0]).label
        # for LEFT, by construction, no need to check whether S0 has all dependents already
        elif ARCHYBRID_LEFTARC_is_legal(c) and gold_dg.get_such_dependency(c.B[0], c.S[0]):
            t = 'LEFTARC'
            nb_dep_to_add[c.B[0]] -= 1
            l = gold_dg.get_such_dependency(c.B[0], c.S[0]).label
        elif ARCHYBRID_SHIFT_is_legal(c):
            t = 'SHIFT'
        else:
            #sys.stderr.write("WARNING: no gold sequence found for tree\n")
            return (None, None, None)

        transitions.append( (t,l) )

        # application of transition to get new configuration:
        applyfuncname = 'ARCHYBRID_'+t+'_'+'apply'
        if keep_configs:
            newc = deepcopy(c)
            #print("NEWC after applying %s :" % t)
            #print(newc)
            eval(applyfuncname)(newc, l)
            others.append(newc)
            c = newc
        else:
            eval(applyfuncname)(c, l)
                        
    return (transitions, c0, others)

# ------------------------------------------- #
# generic classes and methods, independently
# of transition system
# ------------------------------------------- #

class Config:
    def __init__(self, buff, stack, sentence):
        self.B = buff
        self.S = stack
#        self.dg = LightDepGraph(sentence=sentence)
        self.dg = DepGraph(sentence=sentence, check_no_cycle=False)

    def get_form(self, lidx):
        if lidx == DUMMY_ROOT_LIDX:
            return 'ROOT'
        return self.dg.lexnodes[lidx].form
    
    def __str__(self):
        return '[ ' + ' '.join([ self.get_form(x) for x in reversed(self.S)]) + '] ' + ' '.join([self.get_form(x) for x in self.B[:4]]) + "\n" + self.dg.to_string_conll() 

    def full_str(self):
        return '[ ' + ' '.join([ str(x) for x in reversed(self.S)]) + '] ' + ' '.join([str(x) for x in self.B]) + "\n" + self.dg.to_string_conll() + "\n"
        


class TransitionParser:
    
    def __init__(self, system):

        self.name = system

        # les methodes provenant du jeu de transitions choisi
        for s in ['get_initial_config', 'is_config_terminal', 'get_static_gold_transitions', 'get_legal_and_correct_transitions']:
            self.__dict__[s] = eval(system + '_' + s)
#        self.get_initial_config = eval(system + '_get_initial_config')

#        self.is_config_terminal = eval(system + '_is_config_terminal')

#        self.get_static_gold_transitions = eval(system + '_get_static_gold_transitions')
        
        self.transitions = {} # key = transition name, val = Transition instance

        for (transname, is_labeled) in SYSTEM2TRANS[system]:
            self.transitions[transname]= Transition(system, transname, is_labeled)      

    def is_transition_applicable(self, c, transname):
        m = self.name + '_' + transname + '_is_legal'
        #print(m)
        return eval(m)(c)
    
    def apply_transition(self, c, transname, label):
        self.transitions[transname].apply_method(c, label)
        
    def apply_trans_sequence(self, c, seq):
        """ apply to config c a transition sequence seq (transname/label pairs) """
        for (transname, label) in seq:
            print("TRANSITION: "+transname)
            # apply transition t to config c
            self.transitions[transname].apply_method(c, label)
            print(c)
            #self.apply_transition(c, transname)
        return c.dg

    def yield_and_apply_next_static_gold_config(self, gold_dg):
        """ yield next pair (configuration c, transition t) according to static oracle from gold parse,
            in which t is the gold transition to apply on c,
            and applies it
        """
        (transitions, c, others) = self.get_static_gold_transitions(gold_dg, keep_configs=False)
        if transitions:
            for (t, l) in transitions:
                yield( (c, t, l) )
                self.transitions[t].apply_method(c, l)
        else:
            #sys.stderr.write("WARNING: no gold sequence found for tree\n")
            yield(None,None,None)

    def is_transition_correct(self, c, gold_dg, t, l):
        """ whether the transition t has cost 0 given the current config f and the gold_dg """
        m = self.name + '_' + t + '_unlab_cost'
        cost = eval(m)(c, l, gold_dg)
        return (cost == 0)

    def get_legal_transitions(self,c):
        legal_trans = []
        for transname in self.transitions.keys():
            m = self.name + '_' + transname + '_is_legal'
            if eval(m)(c):
                legal_trans.append(transname)
        return legal_trans
    
    def get_correct_transitions(self, c, gold_dg):
        """ correct in the sense of Goldberg and Nivre 2013 (with cost 0 given current tree and gold tree) 
        OBSOLETE: more efficiently done in method <SYSTEM>_get_legal_and_correct_transitions
        """
        correct_trans = []
        for transname in self.transitions.keys():
            m = self.name + '_' + transname + '_unlab_cost'
            cost = eval(m)(c, gold_dg)
            if cost == 0:
                correct_trans.append(transname)
        return correct_trans
    
    def greedy_parse(self, gold_dg, transition_classifier, indices):
        """
        greedily parse sentence: keep applying the best scoring allowed transition for the current configuration until terminal config is reached 
        """
        lidxs = gold_dg.sorted_dependent_lidx()
        sentence = [gold_dg.get_lexnode(x).form for x in lidxs]

        (w_ids, p_ids, l_ids) = get_sequence_feats_for_whole_sentence(gold_dg, indices)
        # don't use lemma ids for now
        
        w_ids = torch.LongTensor( w_ids )
        p_ids = torch.LongTensor( p_ids )
        
        embeddings_of_sentence = transition_classifier.compute_contextualized_embeddings( w_ids, p_ids )
        
        c = self.get_initial_config(sentence)
        #print(c)
        while not(self.is_config_terminal(c)):

            #x = torch.LongTensor(config_to_feat_vect(c, indices, gold_dg))

            # positions (lidxs) of the focused elements given the current config
            focused_lidxs = get_focused_elements(c, feats_description=FEATS1)

            # predicted transitions
            (sorted_scores, sorted_transition_ids) = transition_classifier.predict(embeddings_of_sentence, focused_lidxs)
            #print sorted_scores
            #print sorted_transition_ids

            # apply the first predicted transition that is allowed on current config
            for t_id in sorted_transition_ids:
                (t,l) = indices.i2c[t_id.item()]
                if self.is_transition_applicable(c, t):
                    #print("APPLICATION of %s, %s" % (t, str(l)))
                    self.apply_transition(c, t, l)
                    break
            else:
                print("NO APPLICABLE TRANSITION!")
                print(c)
                break
            #print("transition %s predicted with score %f" % (t, sorted_scores[t_id]))
            #print(c)
        return c.dg

    
class Transition:
    def __init__(self, system, name, is_labeled):
        """ transition type, without label """
        self.system = system
        self.name = name
        self.is_labeled = is_labeled
        
        self.condition_method = eval(system + '_' + name + '_is_legal')
        self.apply_method = eval(system + '_' + name + '_apply')


if __name__ == "__main__":
    sentence = 'le chat gris boit du lait'.split(' ')
    
    system = TransitionParser('ARCSTANDARD')
    seq = ['SHIFT', 'LEFTARC', 'SHIFT', 'RIGHTARC', 'SHIFT', 'LEFTARC', 'SHIFT', 'SHIFT', 'LEFTARC', 'RIGHTARC', 'RIGHTARC']
    
    c = system.get_initial_config(sentence)
    print("INITIAL CONFIG:")
    print(c)
    system.apply_trans_sequence(c, seq)



            
    
