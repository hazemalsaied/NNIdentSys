#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

"""
Parser that builds dependency graphs :
from conll format

@author : Marie Candito
"""

from DepGraph import *
import sys

def read_depparses_conll_stream(stream, check_no_cycle=False, multiple_govs=None, infer_feats_from_tagset=False, add_verbal_information=False, multiple_govs_sep="|", interpret_frame_occs=False, deep_and_surf='surf'):
    """ Reads the given stream, and parses depgraphs one by one
    @param stream : input stream in conll format
    @param multiple_govs : if set to 'disj' : input contains alternative governors (disjunction of governors: for parse revision)
                           if set to 'conj' : input contains multiple governors (conjunction of governors: for deep syntax)
                           (column 7 = list of gov lidx, column 8 : list of labels)
    @param : multiple_govs_sep : separator used for the list of gov lidx / labeld
    @param : infer_feats_from_tagset : whether to infer features from the (fine) tag (mood, coarsecat...) (default=False)
    @param : deep_and_surf : whether the input conll is a the surf tree, the deep graph, or both
    @return : a generator over <DepParse> instances
    """
    line = stream.readline()
    sentence = []
    sentnum = 1
    while line:
        line = line[0:-1]
        line = line.strip(' \t')
        if len(line) > 0:
            sentence.append(line)
        else:
            yield read_depparse_conll(sentence, sentnum, check_no_cycle, multiple_govs, infer_feats_from_tagset, add_verbal_information, multiple_govs_sep, interpret_frame_occs, deep_and_surf)
            sentnum = sentnum + 1
            sentence = []
        line = stream.readline()

def read_depparse_conll(sentence, sentnum=-1, check_no_cycle=False, multiple_govs=None, infer_feats_from_tagset=False, add_verbal_information=False, multiple_govs_sep="|", interpret_frame_occs=False, deep_and_surf='surf'):
    """ Builds a DepParse out of list of lines : each line is in conll format, the whole lines form a sentence
    @precondition: the sentence argument contains exactly the lines corresponding to one depgraph in conll format
    @return : a DepParse instance

    deep_and_surf: use 'both' to load in same structure both surf tree and deep graph
                       'surf' for the surf tree only
                       'deep' for the deep graph only
    """
    # pointless to ask for deep_and_surf and not read multiple_govs
    if deep_and_surf in ['both', 'deep']:
        multiple_govs = 'conj'
        multiple_govs_sep = '|'
    if multiple_govs:
        check_no_cycle = False
    depgraph = DepGraph(check_no_cycle=check_no_cycle, deep_and_surf=deep_and_surf)
    sentid = sentnum
    for line in sentence:
        if line.startswith('#'):
            line = line.replace(' ','')
            if line.startswith('#sent_id='):
                sentid = line[9:]
            elif line.startswith('#sentid='):
                sentid = line[8:]
            continue
        cols = line.split('\t')
        dep_lidx = int(cols[0]) -1
        dep_form = cols[1]
        dep_lemma = cols[2]
        coarsecat = cols[3]
        cat = cols[4]

        # PHEAD and PLABEL
        if len(cols) > 8:
            pgovlidx = cols[8]
            plabel = cols[9]
            if pgovlidx == '_':
                pgovlidx = None
                plabel = None
            else:
                pgovlidx = int(pgovlidx) - 1
        else:
            pgovlidx = None
            plabel = None

        # HEAD and LABEL columns are 7th and 8th
        # if input format contains alternative/conjunctive governors 
        if multiple_govs:
            govs = map(lambda x: x - 1, map(int, cols[6].split(multiple_govs_sep) ) )
            labels = cols[7].split(multiple_govs_sep)
            # si deeponly, on ignore la premiere dep si S: (surf only)
            if (deep_and_surf == 'deep') and labels[0].startswith('S:'):
                labels = labels[1:]
                govs = govs[1:]
                # si le noeud n'est plus connecté => connexion à la DUMMY_ROOT
                if len(govs) == 0:
                    govs = [DUMMY_ROOT_LIDX]
                    labels = ['ignored']
            gov_lidx = govs[0]
            dep_label = labels[0]
            # si arbre surf demandé, on enlève les préfixes, et on ne garde que le label surfacique
            if deep_and_surf == 'surf' and ':' in dep_label:
                dep_label = dep_label.split(':')[1]
        else:
            gov_lidx = int(cols[6]) -1
            dep_label = cols[7]
        if dep_lemma == '_': dep_lemma = None
        # add governor, though form unknown yet
        if gov_lidx != DUMMY_ROOT_LIDX and gov_lidx not in depgraph.lexnodes:
            governor = LexicalNode('', gov_lidx)
            depgraph.add_lexnode(governor)
        if dep_lidx not in depgraph.lexnodes and dep_lidx != DUMMY_ROOT_LIDX:
            d = LexicalNode(dep_form, dep_lidx, cat=cat, features={'coarsecat':coarsecat, 'lemma':dep_lemma})
            depgraph.add_lexnode(d, gov_lidx, dep_label, pgovlidx, plabel)
        else:
            # update features
            d = depgraph.get_lexnode(dep_lidx) 
            d.set_feature('form',dep_form)
            d.set_feature('coarsecat',coarsecat)
            d.set_feature('cat',cat)
            d.set_feature('lemma',dep_lemma)
            depgraph.add_dep_from_lidx(gov_lidx, dep_lidx, dep_label, pgovlidx, plabel)
        # features appearing in column 6 of the conll format
        # are simply added as features to the node
        feats = cols[5].split('|')
        for feat in feats:
            if feat == '_': continue
            try:
                (attr,val) = feat.split('=', 1)
            except: 
                print >> sys.stderr, line
                print >> sys.stderr, feat
            if attr != '':
                d.set_feature(attr, val)
        # if multiple governors supplied
        # if disjunctive : store them in specific features of the dependent node
        # if conjunctive : add the corresponding dependencies
        if multiple_govs == 'disj':
            d.set_feature('other_govs', govs[1:])
            d.set_feature('other_labels', labels[1:])
        # debug 31/01/2018: on ne stocke pas les multiple govs si on demande les arbres surf
        elif multiple_govs == 'conj' and deep_and_surf != 'surf':
            for (i, gov_lidx) in enumerate(govs[1:]):
                dep_label = labels[i+1]
                gov = depgraph.get_lexnode(gov_lidx)
                if not gov:
                    gov = LexicalNode('', gov_lidx)
                    depgraph.add_lexnode(gov)
                #print "AJOUT SUPPLEMENTAIRE de", dep_label, gov_lidx, dep_lidx, govs, labels
                #nb: le pgovlidx et plabel est tjrs le meme...
                depgraph.add_dep( Dep(gov_lidx, d.lidx, dep_label, pgovlidx, plabel) )
    #print depgraph.to_string_conll()
    if infer_feats_from_tagset:
        depgraph.interpret_fine_tagset()
    if add_verbal_information:
        depgraph.add_verbal_information()
    if interpret_frame_occs:
        depgraph.interpret_frame_occs(compute_role_head=True)
    # GROS HACK si deep_and_surf vaut "deep" : on charge tout de meme aussi le "deep_and_surf" au cas où on a tout de meme besoin des chemins surf...
    
    if deep_and_surf == 'deep':
        depparse_surf = read_depparse_conll(sentence, sentnum, multiple_govs, infer_feats_from_tagset, add_verbal_information, multiple_govs_sep, interpret_frame_occs, deep_and_surf='surf')
        return DepParse(sentid=sentid, depgraph=depgraph, surf_dg = depparse_surf.depgraph)

    return DepParse(sentid=sentid, depgraph=depgraph)

                
if __name__ == "__main__":
    import sys
    import cProfile
    def conll_2_conll():
#        for depparse in read_depparses_conll_stream(sys.stdin, multiple_govs='conj', multiple_govs_sep="|", deep_and_surf=True):
        for depparse in read_depparses_conll_stream(sys.stdin, multiple_govs='conj', multiple_govs_sep="|", deep_and_surf='deep'):
            if depparse != None:
                sys.stdout.write(depparse.depgraph.to_string_deep_conll())
    conll_2_conll()
    #cProfile.run('conll_2_conll()')
