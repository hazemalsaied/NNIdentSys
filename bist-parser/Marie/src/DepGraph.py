#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Marie Candito : 
# Dependency Graph structure
"""

# modifications from other DepGraph:
# add_dep_from_lidx : label has defaultvalue
# DepGraph can be created from rich conll, but also from bare list of tokens (sentence keyword)

from collections import defaultdict
from parser_constants import *
from utils import TagSetMap_ftb4_ftbi
from utils import TagSetMap_ftb4_ftbmin
import utils
import sys
import re

# linear index for dummy root of the graph
DUMMY_ROOT_LIDX = -1
# name of the dependency between nodes and dummy root
DUMMY_DEP_LABEL = 'root'
# string for the dependency name when underspecified
UNK_DEP_LABEL = 'dep'
# name of the dependency when the correct governor is unknown
UNK_GOV_DEP_LABEL = 'missinghead'

# to compute syntactic head of role fillers or of triggers
PRIORITARY_CATS=["A", "V", "N", "ADV", "PRO", "CL"]
PRIORITARY_LABELS=['root','ats','ato']


verbs = set(['V', 'VIMP', 'VINF', 'VPP', 'VPR', 'VS'])

FTB4DICT = utils.tagfixer2tagdict(utils.ftb4_fixer())
# hack
FTB4DICT['VPP']['m'] = 'pstpart'
FTB4DICT['VPR']['m'] = 'prtpart'


#-------------------------------------------------
# partie déplacée de do_decorate_depgraphs_with_frames 
# (cf. nécessaire ici pour construction du lemme d'une instance de frame annotée)
#-------------------------------------------------
# PB : pb des lemmes des composants, dans le cas de composés discontinus...
# => recuperation de la liste exhaustive des couples forme+lemme ayant ce pb
# => puis hack spécifique à ces cas !!!
# grep -v NPP ../corpus/ftb/ftb-2012/ftb-2012-v09-expandedcpd.manual_se.conll|cut -f2,3|egrep '[_]'|egrep '[a-zA-Z]'|sort -u
BuggedLemmas = {
('Font',	'faire_défaut') : 'faire',
('Sans',	'sans_compter') : 'sans',
('afin',	'afin_de') : 'afin',
('afin',	'afin_qu\'') : 'afin',
('alors',	'alors_que') : 'alors',
('au',	'gràce_à') : 'à',
('augurer',	'laisser_augurer') : 'augurer',
('aura',	'avoir_lieu') : 'avoir',
('aux',	'grâce_à') : 'à',
('comme',	'comme_si') : 'comme',
('compte',	'rendre_compte') : 'compte',
('compte',	'tenir_compte') : 'compte',
('d\'',	'afin_de') : 'de',
('de',	'afin_de') : 'de',
('de',	'à_cause_de') : 'de',
('dues',	'dues') : 'du',
('due',	'due') : 'du',
('en_matière_d\'',	'en_matière_de') : 'en_matière_de',
('face',	'faire_face') : 'face',
('fait',	'faire_face') : 'faire',
('fait',	'faire_état') : 'faire',
('font',	'faire_preuve') : 'faire',
('grâce',	'grâce_à') : 'grâce',
('laisse',	'laisser_augurer') : 'laisser',
('lieu',	'avoir_lieu') : 'lieu',
('lieu',	'tenir_lieu') : 'lieu',
('mais',	'mais_aussi') : 'mais',
('net',	'net_de') : 'net',
('par',	'par_de') : 'par',
('plus',	'plus_tard') : 'plus',
('preuve',	'faire_preuve') : 'preuve',
('que',	'alors_que') : 'que',
('que',	'que_si') : 'que',
('radios',	'radio_privée') : 'radio',
('rend',	'rendre_compte') : 'rendre',
('rendent',	'rendre_compte') : 'rendre',
('siens',	'le_sien') : 'sien',
('tient',	'tenir_compte') : 'tenir',
('tient',	'tenir_lieu') : 'tenir',
('à',	'grâce_à') : 'à',
('à',	'à_faire') : 'à',
('état',	'faire_état') : 'état',
}

# HACK MOCHE liste de lemmes de composés que l'on n'arrive pas bien à recalculer
# TODO : REPRENDRE TOUT LE TRAITEMENT cf. certains cas sont inexpliqués!!!
HackedCompoundLemmas = {
'à mon avis' : 'à son avis',
'chanter le louange' : 'chanter les louanges',
'prendre son source' : 'prendre sa source',
'trouver son source' : 'trouver sa source',
'faire son preuve' : 'faire ses preuves',
'faire son course' : 'faire ses courses',
'savoir que' : 'sachant que',
'renvoyer le balle' : 'renvoyer la balle',
'ordre de jour' : 'ordre du jour',
'prendre le parole' : 'prendre la parole',
'être donner' : 'étant donné',
'prendre son source' : 'prendre sa source',
'trouver son source' : 'trouver sa source',
'voir que' : 'vu que',
'ce être pourquoi' : 'c\'est pourquoi',
'contre - attaque' : 'contre-attaque',
# voir pb "étant donné", c'est à dire, pourquoi lemme différent pour la 8440 et la 6438
}
Lemma2NormalizedLemma = {
    'duquel' : 'de',
    'auquel' : 'à'
}

ElidedForm2Form = {
    "d'" : "de",
    "qu'" : "que",
    "n'" : "ne", 
    "c'" : "ce",
    "jusqu'" : "jusque",
    "s'" : "se",
    "l'" : "le",
}

# correspondances ftbcat vers lexiconcat
ftbcat2lexiconcat = {"NC" : "n", "NPP" : "n",
                     "VPP" : "v", "V" : "v", "VINF" : "v", "VPR" : "v", "VS" : "v", "VIMP" : "v",
                     "DET" : "det",
                     "ADJ" : "a",
                     "ADV" : "adv",
                     "ADVWH" : "adv", # on fusionne avec adv
                     "P" : "prep", 
                     "P+D":"prep", #, P+D "grâce_à"
                     "PRO":"prep", # PRO juste pour "d'où"
                     "CS" : "conj",
                     # ajout marie
                     "ET" : "et",
                     # ajout marie
                     "CC" : "cc",
                     "P+PRO" : "prep_prorel"}

#blindage marie: si cat inconnue de ftbcat2lexiconcat => on laisse la cat syntaxique
def get_lexicon_cat(syntcat):
    if syntcat in ftbcat2lexiconcat:
        return ftbcat2lexiconcat[syntcat]
    return syntcat

def normalize_corpus_lemma(lemma, form):
    """ le lemme du "se" ds les corpus est "le/lui", mais certaines occurrences sont "se" ou "nous" "vous"... 
    plus des pbs sur les lemmes de composés discontinus """
    if lemma == None:
        return ''
    if lemma in ['se', 'nous', 'vous']:
        return 'le/lui'
    if (form, lemma) in BuggedLemmas:
        return BuggedLemmas[ (form, lemma) ]
    if lemma in Lemma2NormalizedLemma:
        return Lemma2NormalizedLemma[lemma]
    # ici normalisation a faire des lemmes composés merged du ftb : mais pas urgent cf. pour l'instant on travaille sur le expanded ftb, les merged ne sont que ds le sequoia, dans lequel les lemmes sont bien normalisés
    if lemma.startswith('A_'):
        return 'à'+lemma[1:]
    return lemma

def normalize_corpus_form(lemma, form, ftb4_cat):
    # debug 21/08/2014 : le .lower() doit etre apres la gestion du A en debut de phrase, valant à...
    form = form.replace('À','à').replace('É','é').replace('Ê','ê').replace('Ù','ù').replace('È','è').replace('Ô','ô').replace('Î','î')
    if form == 'A' and ftb4_cat == 'P':
        return 'à'
    form = form.lower()
    if form == 'etant':
        return 'étant'
    if form in ElidedForm2Form:
        return ElidedForm2Form[form]
    return form

# marie : copié de do_decorate_depgraphs_with_frames...
def mergedlemma2lexiconlemma(merged_lemma):
    # lexicon lemma : "d'autant plus que" "au-delà de"
    # merged lemma : la meme chose, mais avec un "_" à la place d'un espace "d'_autant_plus_que" "au-delà_de"
    # user lemma = merged lemma mais potentiellement avec des erreurs
    # mechanic lemma : la sequence de lemmes de chaque composant, séparés par "-" : "de_autant_plus_que" "à_-_delà_de"

    #marie debug 22/9/2015 : ds le lexique, le s' est collé...
    #return merged_lemma.replace('_',' ')
    merged_lemma = merged_lemma.replace('_',' ').replace("' ", "'")
    #marie debug 27/04/2016 : quelques lemmes composes récalcitrants, à la hâche
    if merged_lemma in HackedCompoundLemmas:
        return HackedCompoundLemmas[merged_lemma]
    return merged_lemma
    

class RoleFillerOcc:
    def __init__(self, depgraph, name, idr, idf, node_list, flag_list):
        """
        A role filler occurrence
        @param depgraph : DepGraph
        @param name : role name
        @param idr : role id
        @param idf : frame instance id
        @param node_list : list of role filler Lexnodes
        @param flag_list : list of role filler flags
        """
        self.dg = depgraph
        self.name = name
        self.idr = idr
        self.idf = idf
        self.nodes = node_list
        self.flags = flag_list
        self.syntheadlidx = -1
        self.semheadlidx = -1
        self.synthead = "" # syntactic head lemma
#        self.semhead = "" # semantic head lemma

    def get_id(self):
        return self.idf + '.' + self.idr

    def add_node(self, node):
        """
        Adds a Lexnode to the list of role filler Lexnodes
        @param node : Lexnode
        """
        self.nodes.append(node)

    def add_feature_to_rolefillerocc(self, lexicalnode, feat):
        """ ajout du trait feat au trait role de lexicalnode 
        très moche, la gestion du trait role en fonction des rolefillerocc est a reprendre complètement """
        role_feature_value = lexicalnode.get_feature_value('role')
        id = self.idf + '.' + self.idr
        roles = role_feature_value.split(",")
        for i,role in enumerate(roles):
            splitted_role = role.split("#")
            (frame_id, role_id) = splitted_role[0].split('.')
            if splitted_role[0] == id:
                if feat not in role:
                    roles[i] += '#'+feat
                break
        lexicalnode.set_feature('role', ','.join(roles))

    def is_null_instantiation(self):
        return (len(self.nodes) == 0)
            
    def compute_syntactic_head(self):
        """
        Computes syntactic head(s) of a RoleFillerOcc, fills self.synthead
        @param prioritary_cats : list of sorted categories to choose a head
        @return a list of syntactic head node forms

        NB: works on the surface tree, otherwise the head notion is not well-defined
        """
        # skip s'il s'agit d'un role filler vide (null instantiation)
        if self.nodes == []:
            return

        # si role filler reduit à un seul noeud: trivial
        if len(self.nodes) == 1:
            head_node = self.nodes[0]
        else:
            #print "COMPUTE_SYNTACTIC_HEAD", self.idf, self.idr, self.name
            synthead_list = []
            role_filler_lidxs = [node.lidx for node in self.nodes]
        
            # recuperation des racines de chaque sous-arbre non connecté du rôle:
            # pour chaque noeud du role
            for node in self.nodes:
                head_dep_label = None
                # si le noeud n'est pas une racine
                if node.lidx in self.dg.dependent2deps:
                    # si le gouverneur de node n'est pas ds role_filler_lidxs
                    for dep in self.dg.dependent2deps[node.lidx]:
                        if dep.governor.lidx not in role_filler_lidxs:
                            # on stocke le label fonctionnel de cette tête
                            head_dep_label = dep.label
                else:
                    head_dep_label = 'root'
                if head_dep_label:
                    synthead_list.append((node,head_dep_label))
        
            # une seule tête
            if len(synthead_list) <= 1:
                head_node = synthead_list[0][0]
            # multihead
            else:
                h_list = [] # contiendra [(mot1, cat1, label1), (mot2, cat2, label2), ...]
                h2node = defaultdict()
                for (node, head_dep_label) in synthead_list:
                    if node.get_feature_value("mwehead"): # mots composés
                        coarsecat = TagSetMap_ftb4_ftbmin[node.get_feature_value("mwehead")]
                        t = (node.form, coarsecat, head_dep_label, node.lidx)
                    else:
                        t = (node.form, node.coarsecat, head_dep_label, node.lidx)
                    h_list.append(t)
                    h2node[t] = node

                def my_index(l, elt):
                    try:
                        return l.index(elt)
                    except ValueError:
                        return 100
                # marie debug 7/3/2017: meilleur calcul de la tete
                def sort_heads(h1, h2):
                    #print "sort_heads", h1, h2
                    (form1, cat1, label1, lidx1) = h1
                    (form2, cat2, label2, lidx2) = h2
                    l1 = my_index(PRIORITARY_LABELS, label1)
                    l2 = my_index(PRIORITARY_LABELS, label2)
                    c1 = my_index(PRIORITARY_CATS, cat1)
                    c2 = my_index(PRIORITARY_CATS, cat2)
                    if l1 + l2 < 200:
                        #print "L1L2", cmp(l1,l2)
                        return(cmp(l1,l2) or cmp(lidx1,lidx2))
                    if c1 + c2 < 200:
                        #print "C1C2", cmp(c1,c2)
                        return(cmp(c1,c2) or cmp(lidx1,lidx2))
                    return cmp(lidx1,lidx2)

                sorted_heads = sorted(h2node.keys(), sort_heads)
                sorted_heads_no_ponct = [ x for x in sorted_heads if x[1] != 'PONCT' ]
                if sorted_heads_no_ponct:
                    head_node = h2node[sorted_heads_no_ponct[0]]
                else:
                    head_node = h2node[sorted_heads[0]]

        self.synthead = head_node.form
        self.syntheadlidx = head_node.lidx
            # ajout trait synthead
            # marie debug 17/05/2016: le trait role contient tous les roles!
            #head_cat2node[buff[0]].set_feature("role", head_cat2node[buff[0]].get_feature_value("role")+"#synthead=y")
        self.add_feature_to_rolefillerocc(head_node, 'synthead=y')

        return self.synthead

    def __str__(self):
        """
        Format : nom_du_role#flags_du_role#la_sequence_de_tokens_du_role_filler#tete_syntaxique
        """
        s = "#".join([self.name, ",".join(self.flags), " ".join([node.form for node in self.nodes]), self.synthead])
        return s


class FrameOcc:
    def __init__(self, depgraph, name, sentid, idf, target_node_list, flag_list):
        """
        A frame occurrence
        @param depgraph : DepGraph
        @param name : frame name
        @param sentid : sentence id
        @param idf : frame id
        @param target_node_list : list of target Lexnodes
        @param flag_list : list of frame flags
        """
        self.dg = depgraph
        self.name = name
        self.sentid = sentid
        self.idf = idf
        self.target_node_list = sorted(target_node_list, lambda x,y: cmp(x.lidx, y.lidx))

        self.id2filler = {} # {role_id : RoleFillerOcc}

        self.target_cat = self.compute_cat_of_target()
        self.target_head = self.compute_head_of_target()
        self.target_lemma = '' # calculé infra

        # reconstitution de la chaîne correspondant au trigger lemmatisé
        # avec reconnaissance et distinction des formes en "se" et "s'"
        target_lemma_list = []
        nb_components = len(self.target_node_list)
        # pour les prep composees : plus facile de passer par les formes
        # voir egalement ensuite mergedlemma2lexiconlemma, qui utilise HackedCompoundLemmas
#        if nb_components > 1 and (self.target_node_list[0].cat in ['P','P+D'] or self.target_cat == 'conj'):
        if nb_components > 1 and (self.target_node_list[0].cat in ['P','P+D'] or self.target_cat in ['conj','cc','adv']):
            target_lemma_list = [ x.form.lower().replace('É','é') for x in self.target_node_list ]
        else:
            target_lemma_list = [ normalize_corpus_lemma(x.get_feature_value("lemma"),x.form.lower()) for x in self.target_node_list ]
        #print target_lemma_list
        if nb_components > 1:
            first = target_lemma_list[0]
            if first in ["À", "a"]:
                target_lemma_list[0] = "à"
            if first.startswith("le/lui"):
                for c in ['à','a', 'e', 'i', 'o', 'u', 'y',"é",'ê']:
                    if target_lemma_list[1].startswith(c):
                        target_lemma_list[0] = "s'"
                        break
                else:
                    target_lemma_list[0] = "se"
            for i, lemma in enumerate(target_lemma_list):
                #print i, lemma
                if i < len(target_lemma_list)-1:
                    if lemma in ['le','de'] or lemma.endswith('que'):
                        for c in ['à','a', 'e', 'i', 'o', 'u', 'y',"é",'ê','h']:
                            if target_lemma_list[i+1].startswith(c):
                                target_lemma_list[i] = lemma[:-1]+"'"
                                break
            last_tok = {"qu'" : "que", "d'" : "de", "au" : "à", "aux" : "à", "des" : "de", "du" : "de"}
            for tok, new_tok in last_tok.items():
                if target_lemma_list[-1].endswith(tok):
                    target_lemma_list[-1] = new_tok

        #marie debug : les lemmes contenant des "_" (cf. sequoia) doivent etre rendus compatibles avec lemme du lexique (avec espaces)
        #self.target_lemma = " ".join(target_lemma_list)
        self.target_lemma = mergedlemma2lexiconlemma(" ".join(target_lemma_list))

        # marie: on stocke le target_lemma comme trait sur premier token du target
        self.target_node_list[0].set_feature('targetlemcat', '"'+self.target_lemma + '.' + self.target_cat+'"')

        #if len(self.target_node_list) > 1:
        #    print("MULTI LEMMA %s, HEAD=%s" % (self.target_lemma, self.target_head.lemma))
        self.flags = flag_list

    def __eq__(self, other):
        # note : we know that the target_node_list is already sorted
        lidxs = [ x.lidx for x in self.target_node_list ]
        other_lidxs = [ x.lidx for x in other.target_node_list ]

        return self.name == other.name and (lidxs == other_lidxs)

    # TODO: refusionner avec compute_syntactic_head de RoleFillerOcc
    def compute_head_of_target(self):
        """
        Computes syntactic head of a the target
        use of PRIORITARY_CATS : list of sorted categories to choose a head
        @return the best synthead node of the target
        """
        # si target n'a qu'un composant
        if len(self.target_node_list) == 1:
            return self.target_node_list[0]
        
        synthead_list = []
        target_lidxs = [node.lidx for node in self.target_node_list]
        #print "COMPUTE_HEAD_OF_TARGET", self.idf, self.name, target_lidxs
        # recuperation des racines de chaque sous-arbre non connecté du target:
        # pour chaque noeud du target
        for node in self.target_node_list:
            head_dep_label = None
            # si le noeud n'est pas une racine
            if node.lidx in self.dg.dependent2deps:
                # si le gouverneur de node n'est pas ds role_filler_lidxs
                for dep in self.dg.dependent2deps[node.lidx]:
                    #if dep.governor.lidx not in target_lidxs:
                    # 9.02.2018: on ignore les noeuds ecartés ds graphe deep (noeuds void non connectés)
                    if dep.label != 'ignored' and dep.governor.lidx not in target_lidxs:
                        # on stocke le label fonctionnel de cette tête
                        head_dep_label = dep.label
            else:
                # TODO A DEBUGGER ICI, cas des S:, "faire" ok mais "se" non...
                # HACK
                # 9.02.2018: finalement pas une bonne idee de booster "faire" comme tête en cas de trigger composé
                if False: # node.lemma == 'faire':
                    head_dep_label = 'root'
                else:
                    head_dep_label = 'dummy' 
            if head_dep_label:
                synthead_list.append((node,head_dep_label))

        # une seule tête
        if len(synthead_list) == 1:
            return synthead_list[0][0]
        # pas de tete trouvee = les composants sont un ou des noeuds void non connectes
        # => on prend le premier
        elif len(synthead_list) == 0:
            return self.target_node_list[0]
        # multihead
        else:
            h_list = [] # contiendra [(mot1, cat1, label1), (mot2, cat2, label2), ...]
            h2node = defaultdict()
            for (node, head_dep_label) in synthead_list:
                if node.get_feature_value("mwehead"): # mots composés
                    coarsecat = TagSetMap_ftb4_ftbmin[node.get_feature_value("mwehead")]
                    t = (node.form, coarsecat, head_dep_label, node.lidx)
                else:
                    t = (node.form, node.coarsecat, head_dep_label, node.lidx)
                h_list.append(t)
                h2node[t] = node

            def my_index(l, elt):
                try:
                    return l.index(elt)
                except ValueError:
                    return 100
            # marie debug 7/3/2017: meilleur calcul de la tete
            def sort_heads(h1, h2):
                #print "sort_heads", h1, h2
                (form1, cat1, label1, lidx1) = h1
                (form2, cat2, label2, lidx2) = h2
                l1 = my_index(PRIORITARY_LABELS, label1)
                l2 = my_index(PRIORITARY_LABELS, label2)
                c1 = my_index(PRIORITARY_CATS, cat1)
                c2 = my_index(PRIORITARY_CATS, cat2)
                if l1 + l2 < 200:
                    #print "L1L2", cmp(l1,l2)
                    return(cmp(l1,l2) or cmp(lidx1,lidx2))
                if c1 + c2 < 200:
                    #print "C1C2", cmp(c1,c2)
                    return(cmp(c1,c2) or cmp(lidx1,lidx2))
                return cmp(lidx1,lidx2)

            sorted_heads = sorted(h2node.keys(), sort_heads)
            #print sorted_heads
            sorted_heads_no_ponct = [ x for x in sorted_heads if x[1] != 'PONCT' ]
            if sorted_heads_no_ponct:
                head_node = h2node[sorted_heads_no_ponct[0]]
            else:
                head_node = h2node[sorted_heads[0]]

        return head_node

    def compute_cat_of_target(self):
        """
        calcul de la categorie de type FfnLexicon d'un trigger (compose)
        """
        nodes = self.target_node_list
        first_tok_cat = nodes[0].cat

    # Triggers d'un mot
        if len(nodes) == 1:
            if first_tok_cat == "P+D": # grâce_à
                cat = "prep"
            # marie 7/01/16 : trop brutal, "certains" est parfois det
            # elif first_tok_cat == "DET": # erreur dans treebank 6283 : commerce = DET
            #     for x in nodes:
            #         sys.stderr.write("WARNING : DET on token \"" + str(x.form) + "\"\n")
            #         cat =  "n"
            else:
                cat = get_lexicon_cat(first_tok_cat)
    
    # Triggers de plusieurs mots
        else:

            # triggers commençant par "en", "par" ou "à"
            if nodes[0].form.lower() in ["en", "par", "à"]:
                if nodes[-1].cat == "NC":
                    cat = "adv"
                else:
                    cat = get_lexicon_cat(nodes[-1].cat)
            elif [ x.form.lower() for x in nodes ] == ['dans', 'la', 'mesure', 'où']:
                cat = "conj" # cf. donne sinon prorel
        # triggers commençant par "ne", "que" ou "au" si 2 mots
            elif nodes[0].form.lower() == "ne":
                cat = "v"
            elif nodes[-1].form.lower() == "que":
                cat = "conj"
            elif nodes[0].form.lower() == "au" and len(nodes) == 2: # au courant, au fait
                cat = "a"
            elif first_tok_cat.startswith("V"):
                cat = "v"
            # marie 2/12/2015
            # si mwehead disponible => on prend ça (mais pb , on pourrait utiliser heuristique si fin=prep => prep)
            elif nodes[-1].cat in ['P','P+D']:
                return 'prep'
            elif nodes[0].get_feature_value('mwehead'):
                return get_lexicon_cat(nodes[0].get_feature_value('mwehead'))
        
            else: #cat du dernier mot de la locution
                cat = get_lexicon_cat(nodes[-1].cat)
        #print "COMPUTE_CAT_OF_TARGET=>", cat, 'from', ' # '.join([str(x) for x in nodes])
        return cat

    
    def add_role_filler(self, role_occ):
        """
        Adds a RoleFillerOcc to self.id2filler
        @param role_occ : RoleFillerOcc
        """
        self.id2filler[role_occ.idr] = role_occ
          
    def __str__(self):
        """
        @param multihead_roleocc : True => n'intègre à la chaîne que les role_occ avec plusieurs têtes syntaxiques
        @return string. Format :
        frame sentid trigger_id trigger_lemma trigger_form nom_du_role#flags_du_role#la_sequence_de_tokens_du_role_filler
        """
        target_ids, target_form = [], []
        for target_node in self.target_node_list:
            target_ids.append(str(target_node.lidx + 1))
            target_form.append(target_node.form)

        s = "\t".join([self.name, self.sentid, ",".join(target_ids), self.target_lemma, " ".join(target_form)])
        for role_id, role_occ in sorted(self.id2filler.items(), lambda x,y : cmp(x[1].name, y[1].name)):
            s += "\t" + str(role_occ)
        s += "\n"
        return s

    
class LexicalNode:
    def __init__(self, form, lidx, cat=None, features=None):
        """ Lexical node, identified by lidx (linear index)
        CAUTION : two LexicalNodes with same lidx are considered equal
        Members:
        * form : string for the word / token / whatever
        * lidx : identifier of the node (currently the linear index of the node)
        * cat : morpho-syntactic category, default = None

        Additional members can be added via set_feature

        @param form
        @param lidx
        @param lemma default=None
        @param cat default=None
        @param features : default=None, dictionary for additional member/value pairs

        Usual additional members, when built from conllreader
        * coarsecat : the coarse grained part of speech ("basecat")
        * lemma 
        
        """
        self.lidx = lidx
        self.form = form
        self.cat = cat

        if features != None:
            self.__dict__.update(features)

    def __eq__(self, other):
        return isinstance(other, LexicalNode) and self.lidx == other.lidx

    def __ne__(self, other):
        return not isinstance(other, LexicalNode) or self.lidx != other.lidx

    def __str__(self):
        return str(self.lidx)+" "+self.form+" "+self.cat

    def get_feature_value(self,attr):
        """ Returns the value of the given attribute, if defined,
        otherwise returns None 
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        return None

    def get_gold_gov_lidx(self):
        """ returns the lidx of the gold governor, if available, None otherwise """
        return self.get_feature_value('gglidx')

    def set_feature(self,attr, value):
        """ Sets the value of the given attribute """
        self.__dict__[attr] = value

    def del_feature(self,attr):
        """ deletes the given attribute """
        if attr in self.__dict__:
            del(self.__dict__[attr])

    # For LexicalNodes containing spine
    def add_to_spine(self,label):
        if 'spine' not in self.__dict__ or self.spine == None:
            self.spine = []
        self.spine.append(label)

    def maximal_projection(self):
        return self.spine[-1]

    def first_projection(self):
        if len(self.spine) > 1:
            return self.spine[1]
        else:
            return None

    def x_projection(self,x):
        if len(self.spine) > x:
            return self.spine[x]
        else:
            return None

    def is_dummy_root(self):
        return self.lidx == DUMMY_ROOT_LIDX

    def is_prep(self):
        return self.cat in ['P','P+D']

    def is_prep_det(self):
        """ Returns True if the node corresponds to a prep amalgamated with a determiner
        it relies on the cat (P+D) for simple preps, or on the suffix of the prep, in case of a multi-word preposition """
        if self.cat == 'P+D':
            return True
        m = re.match('_(?:aux?|du|des)$', self.form)
        if m != None:
            return True
        return False

    ## Printing Functions

    def __str__(self):
        return self.to_string_pivot()

    def feature_value_to_string(self,attr,dummy=None,normalize=False):
        a = self.get_feature_value(attr)
        if dummy != None and a == None:
            return dummy
        # if normalize:
        #     if isinstance(a, str):
        #         return encode_const_metasymbols(a)
        #     return encode_const_metasymbols(str(a))
        return str(a)

    def feature_to_string_pivot(self,attr):
        if attr == 'cat':
            pattr = 'pos'
        else: pattr = attr
        return pattr+'('+self.to_string_pivot()+','+self.feature_value_to_string(attr,normalize=True)+')'

    def to_string_pivot(self):
        #return encode_const_metasymbols(self.form)+'~'+str(self.lidx)
        return self.form+'~'+str(self.lidx)

    def to_string_conll(self):
        """ builds the string for the conll format : the first columns for information on the node (not the dependency) """
        # conll ID : the lidx + 1 ...
        vals = [ str(self.get_feature_value('lidx') + 1) ]
        # conll FORM, LEMMA, CPOSTAG, POSTAG
        vals.extend([ self.feature_value_to_string(x,dummy='_')
                      for x in ['form','lemma','coarsecat','cat']])
        # conll FEATS
        feats = []
        # c'est pourri! il faudrait distinguer les attributs in et out
        for attr in sorted([a for a in self.__dict__.keys() if a not in ['form','lemma','coarsecat','cat','lidx','spine','deepmood','fulllemma','voice']]):
# à activer si on veut voir les traits deep        for attr in sorted([a for a in self.__dict__.keys() if a not in ['form','lemma','coarsecat','cat','lidx']]):
            val = self.feature_value_to_string(attr, dummy='')
            if val:
                feats.append(attr+'='+val)
        if len(feats) > 0:
            vals.append('|'.join(feats))
        else:
            vals.append('_')
#        a = self.feature_value_to_string('fctpath',dummy='_')
#        if a != '_':
#            vals.append('fctpath='+a)
#        else:
#            # TODO HERE : dump the original features!
#            vals.append('_')
        return '\t'.join(vals)
        
DUMMY_ROOT = LexicalNode('null',DUMMY_ROOT_LIDX, cat='null')
DUMMY_ROOT.set_feature('lemma', 'null')

# a dependency
class Dep:

    def __init__(self, governor, dependent, label=None, pgovlidx=None, plabel=None):
        """ A dependency
        @param governor : the governor (or head) of the dependency
        @type governor : LexicalNode
        @param dependent : the dependent of the dependency
        @type dependent : LexicalNode
        @param label : the label of the dependency
        @type label : string
        pgovlidx and plabel stand for the lidx of projective governor, and projective label
        """
        self.label = label
        self.governor = governor
        self.dependent = dependent
        if pgovlidx != None:
            self.pgovlidx = pgovlidx
            self.plabel = plabel

    def __eq__(self, other):
        return self.dependent.lidx == other.dependent.lidx and self.governor.lidx == other.governor.lidx and self.label == other.label

    def substitute_surface_by_deep(self, deeplabel, deepgov):
        """ additional pair of governor + label, for alternative deep dependency 
        the original dep is stored in pgovlidx and plabel (correspond to 'phead' et 'plabel' in conll 2006 format)"""
        self.plabel = self.label
        self.label = deeplabel
        self.pgovlidx = self.governor.lidx
        self.governor = deepgov

    def get_feature_value(self,attr):
        """ Returns the value of the given attribute, if defined,
        otherwise returns None 
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        return None

    def feature_value_to_string(self,attr,dummy=None):
        a = self.get_feature_value(attr)
        if dummy != None and a == None:
            return dummy
        return str(a)

    def __eq__(self, other):
        return isinstance(other,Dep) and self.label == other.label and self.governor == other.governor and self.dependent == other.dependent

    def __ne__(self, other):
        return not isinstance(other, Dep) or self.label != other.label or self.governor != other.governor or self.dependent != other.dependent

    def __cmp__(self, other):
        """ !!! CAUTION : comparisons on the dependent lidx only """
        return cmp(self.dependent.lidx, other.dependent.lidx)

    def unlabeled_eq(self, other):
        return self.governor == other.governor and self.dependent == other.dependent

    def is_unlabelled(self):
        return self.label == UNK_DEP_LABEL or self.label == None

    def __str__(self):
        return self.to_string_pivot()

    def label_to_string(self):
        if self.label == None:
            return UNK_DEP_LABEL
        return self.label.lower()

    def to_string_pivot(self):
        return self.label_to_string()+'('+self.governor.to_string_pivot()+','+self.dependent.to_string_pivot()+')'

    def to_string_trace(self):
        """ For tracing purposes : a tab separated string for the dependency, including categories of governor and dependent """
        gov_cat = self.governor.cat if self.governor.cat != None else str(None)
        return '\t'.join( [self.label_to_string(),
                           '(',
                           gov_cat+'~'+self.governor.to_string_pivot(),
                           ',',
                           self.dependent.cat+'~'+self.dependent.to_string_pivot(),
                           ')'] )
                           

    def to_string_conll(self):
        """ one line of the conll format : a line for one token and its dependency to its governor """
        a = self.dependent.to_string_conll()
        # conll HEAD,DEPREL
        head = str(self.governor.lidx + 1)
        label = self.label_to_string()
        # potential canonical label
        canonlabel = self.get_feature_value('canonlabel')
        if canonlabel:
            label += ':'+canonlabel
        a = a + '\t' + head + '\t' + label
        # conll PHEAD, PDEPREL
        pgovlidx = self.get_feature_value('pgovlidx')
        if pgovlidx != None:
            a = a + '\t' + str(pgovlidx + 1) + '\t' + self.feature_value_to_string('plabel','_')
        else:
            # non : si pgovlidx est None => signifie que l'on avait en entree "_" => ce qu'on refournit en sortie
            #a = a + '\t' + head + '\t' + label
            a = a + '\t_\t_'
        return a

class DepGraph:
    """
    A dependency graph
    Restrictions : if check_no_cycle:
    = a directed acyclic graph (no way to add dependencies that create a cycle)

    A dummy node may appear in dependencies, as governor only,
    and it does not appear as node of the graph
    The dummy node has by convention lidx = DUMMY_ROOT_LIDX

    FIELDS:
    lexnodes : a dictionary mapping lidx to LexicalNodes
    deps : a list of Dep
    
    dependent2deps : a dictionary mapping dependent lidx to the list of dependencies with such dependent
    governor2deps : a dictionary mapping governor lidx to the list of dependencies with such governor

    @param check_no_cycle : whether to check whether the dep adds a cycle
    @type : boolean
    @default : True

    """
    def __init__(self, sentence = None, check_no_cycle=True, deep_and_surf='surf'):
        self.lexnodes = {} # dict : key=lidx, val = lexnode
        if sentence:
            for lidx, tok in enumerate(sentence):
                self.lexnodes[lidx] = LexicalNode(tok, lidx)
        self.deps = []
        self.dependent2deps = {}
        self.governor2deps = {}        
        self.frames = [] # FrameOcc list
        self.deep_and_surf = deep_and_surf # 'both', 'deep', 'surf'
        self.check_no_cycle = check_no_cycle

    def get_syntactic_neighbors(self, lidx, length=1, bypass_cats = ['P','P+D','CS'], direction=None):
        """ starts from the node at provided lidx and retrieves syntactic neighbors:
        pairs of (path, lexical item found following that path from start node)
        with path of length <= length (not counting the bypass_cats in lengths)

        In a path, the steps are either all up or all down

        @direction: if 'up' (resp. 'down') => up (resp. down) paths only, 
                    if set to None, both directions will be used
        """
        current_paths = [('', self.get_lexnode(lidx))]
        a = []
        if direction != 'up':
            a = self.__get_syntactic_neighbors(current_paths, 'down', length, bypass_cats)
            if direction == 'down':
                return a
        b = self.__get_syntactic_neighbors(current_paths, 'up', length, bypass_cats)
        return a+b
    
    def __get_syntactic_neighbors(self, current_paths, direction, length, bypass_cats, filter_dep_labels=['ponct']):
        """ recursively augments paths

        @param: current_paths is a list of pairs (path, lexical_node reached following that path)
                Is updated with additional longer paths
        @direction: either 'up' or 'down'
        @length: maximum length of paths (modulo bypass_cats)
        @bypass_cats: nodes whose category is in that list do not count for path length

        @return: pairs of (path, lexical item found following that path from start node)
        """
        new_paths = []

        for (path, node) in current_paths:
            lidx = node.lidx
            if direction == 'down':
                # if dependencies have deep / surface status, then discard "surface only" dependencies (in surf but not in deep)
                steps = [ (path+'/'+'+'+d.label, d.dependent) for d in self.get_dep_by_governor(lidx) if 'deep' not in d.__dict__ or d.deep ]
            else:
                steps = [ (path+'/'+'-'+d.label, d.governor) for d in self.get_dep_by_dependent(lidx) if 'deep' not in d.__dict__ or d.deep ] 
            new_paths += steps

            for (prefix, onode) in steps:
                new_length = length
                if onode.cat not in bypass_cats:
                    new_length = length - 1
                if new_length > 0:
                    # get neighbors with paths of length length - 1
                    paths_ends = self.__get_syntactic_neighbors([('', onode)], direction, new_length, bypass_cats)
                    # concatenate with prefix (/ already added)
                    new_paths += [ (prefix + x[0],x[1]) for x in paths_ends ]
        #print([ (p, str(n)) for (p, n) in new_paths ])
        return new_paths

#    def interpret_frame_occs(self, compute_role_head=False, prioritary_cats=["V*", "ADJ", "N*", "ADV", "PRO"]):
    def interpret_frame_occs(self, compute_role_head=False):
        """
        Interprets frames occurrences from a DepGraph,
        fills self.frames with FrameOcc
        @param compute_role_head : if True, computes RoleFillerOcc syntactic head(s) 
        and adds a feature to the Lexnode, else reads the syntactic head
        @param prioritary_cats : list of sorted categories to choose a head
        """
        frameid2frames = {} # {frame_id : frame_name}
        frameid2role_occ = defaultdict(lambda:defaultdict()) # {frame_id : {role_id:role_occ}}
        frameid2trigger = defaultdict(list) # {frame_id : [trigger_lexnodes]}
        frameid2flags = defaultdict(list)
        sentid = self.lexnodes[0].get_feature_value("sentrk")

#        for lidx, lexnode in self.lexnodes.iteritems():
        for lidx, lexnode in sorted(self.lexnodes.items(), key=lambda x: x[0]):
            frame_feature_value = lexnode.get_feature_value("frame")
            role_feature_value = lexnode.get_feature_value("role")
            # frames
            if frame_feature_value:
                frames = frame_feature_value.split(",")
                for frame in frames:
                    splitted_frame = frame.split("#")
                    frame_id, frame_name = splitted_frame[0], splitted_frame[1]
                    frameid2frames[frame_id] = frame_name
                    frameid2trigger[frame_id].append(lexnode)
                    flags = []
                    for attr_val in splitted_frame[2:]:
                        # rôles non exprimés
                        if attr_val.split("=")[1] in ["UNI", "ENI", "DNI", "LNI"]:
                            role_name, flag = attr_val.split("=")
                            # marie debug 22/9/2015: le idrole ne peut pas etre None, sinon pas possib de stocker plusieurs roles null instantiation
                            #                        => on utilise ds ce cas comme id de role le nom du role
                            #role_occ = RoleFillerOcc(self, role_name, None, [], [flag])
                            role_occ = RoleFillerOcc(self, role_name, role_name, frame_id, [], [flag])
                            frameid2role_occ[frame_id][role_name] = role_occ # role_id = role_name pour les rôles non exprimés 
                        # flags sur les frames
                        elif attr_val.split("=") == "flags":
                            frameid2flags[frame_id] = attr_val_split("=")[1].split("+")
                            
            # rôles exprimés
            if role_feature_value:
                roles = role_feature_value.split(",")
                # si mode calcul des tetes synt : on commence par enlever toutes les marques de tetes synt
                # (remarque : le semhead est une info manuelle, pas calculée, donc à ne pas enlever)
                if compute_role_head:
                    # on garde [0-9] pour le debug
                    n_role_feature_value = re.sub("#[0-9]?synthead=y", "",role_feature_value)
                    lexnode.set_feature('role', n_role_feature_value)
                for role in roles:
                    splitted_role = role.split("#")
                    # marie debug 27/04/2016 : l'id de frame peut avoir une longueur > 1 !
                    #frame_id, role_id = splitted_role[0][0], splitted_role[0][-1]
                    frame_and_role_id = splitted_role[0]
                    (frame_id, role_id) = frame_and_role_id.split('.')
                    role_name = splitted_role[1]
                    feats = splitted_role[2:]

                    # lecture flags et synthead / semhead
                    flag_list = []
                    semheadlidx = -1
                    syntheadlidx = -1
                    synthead = ''
                    for (attr,val) in [ x.split('=') for x in feats]:
                        if attr == "flags":
                            flag_list = val.split("+")
                        elif attr == 'semhead':
                            semheadlidx = lexnode.lidx
                        elif attr == 'synthead':
                            syntheadlidx = lexnode.lidx
                            # TODO HERE : il faudrait calculer la forme composée!
                            # VOIR le calcul du target_lemma ds le constructeur de FrameOcc
                            synthead = lexnode.lemma
                        #else:
                        #    sys.stderr.write("WARNING:unknown feat in role filler occ: "+attr+' = '+val+"\n")

                        # OBSOLETE?? il n'y a plus la forme de la tete
                        #if not compute_role_head: # lecture de la tête # marie: a revoir
                        #if len(feats) > 0 and "=" not in feats[-1]:
                        #     role_occ.synthead.append(feats[-1])
                        
                    # si ce role_id a déjà été rencontré (normalement les flags sont vides, cf. ils sont sur premier noeud du rolefiller)
                    if role_id in frameid2role_occ[frame_id]:    
                        frameid2role_occ[frame_id][role_id].add_node(lexnode)
                    else:
                        role_occ = RoleFillerOcc(self, role_name, role_id, frame_id, [lexnode], flag_list)
                        frameid2role_occ[frame_id][role_id] = role_occ                
                    if semheadlidx > -1:
                        frameid2role_occ[frame_id][role_id].semheadlidx = semheadlidx
                    if syntheadlidx > -1:
                        frameid2role_occ[frame_id][role_id].syntheadlidx = syntheadlidx
                        frameid2role_occ[frame_id][role_id].synthead = synthead
                        

        # création des FrameOcc et ajout à self.frames
        for frame_id, frame_name in frameid2frames.iteritems():
            frame_occ = FrameOcc(self, frame_name, sentid, frame_id, frameid2trigger[frame_id], frameid2flags[frame_id])
            for role_id, role_occ in frameid2role_occ[frame_id].iteritems():
                # marie 25/5/2016: finalement on calcule la tete synt de tous les roles (et eventuellement la tete sem)
                if compute_role_head:
                    role_occ.compute_syntactic_head()
                frame_occ.add_role_filler(role_occ)
            self.frames.append(frame_occ)
            
    
    def add_lexnode(self,lexnode,gov_lidx=None,label=None, pgovlidx=None, plabel=None):
        """ Adds a lexical node to the graph,
        if a governor lidx is supplied (and != None), also adds the relevant dependency between this node
        and the governor
        @param lexnode : the LexicalNode
        @param gov_lidx : the lidx of its governor (default=None)
        @param label : the label of the dependency between lexnode and the supplied governor
        """
        self.lexnodes[lexnode.lidx] = lexnode
        if gov_lidx != None:
            # todo report error if not in self.lexnodes
            g = self.get_lexnode(gov_lidx)
            if g != None:
                self.add_dep( Dep(g,lexnode,label,pgovlidx,plabel) )

    def add_dep(self, dep , check_no_cycle=None):
        """ Adds a Dep in the graph, and updates the indexes accordingly 
        NOTE : the nodes of the Dep are not required to be in the graph, and are not added
        NOTE : if dependent is the dummy node (lidx=DUMMY_ROOT_LIDX) : the dep is not added
        NOTE : check_no_cycle is either provided as argument, either taken from self.check_no_cycle
        NOTE : if check_no_cycle is True, the dep is not added if it creates a cycle
        @param dep : a dependency
        @type : Dep
        """
        if check_no_cycle == None:
            check_no_cycle = self.check_no_cycle
        gov_lidx = dep.governor.lidx
        dep_lidx = dep.dependent.lidx
        # block dependent dummy
        if dep_lidx == DUMMY_ROOT_LIDX:
            return False

        # block cycles
        if check_no_cycle and gov_lidx in self.get_reachable_nodes(dep_lidx):
            #exit("Cycle detected between nodes "+str(gov_lidx)+" and "+str(dep_lidx)+" !!!")
            return False

        # si on a prévenu que le graphe est en deep_and_surf (=compacte), alors on interprete les prefixes S: et D:
        surf = True
        deep = True

        if dep.label.startswith('S:'):
            deep = False
            dep.label = dep.label[2:]
        elif dep.label.startswith('D:'):
            surf = False
            dep.label = dep.label[2:]
        elif dep.label.startswith('I:'):
            # cas tres tordu des arcs intermediaire, ni surf ni profond...
            surf = False
            deep = False
            dep.label = dep.label[2:]

        # si en outre on ne veut que les arcs deep
        if not deep and self.deep_and_surf == 'deep':
            return False
        # si en outre on ne veut que les arcs surf
        if not surf and self.deep_and_surf == 'surf':
            return False

        # en outre, on ne garde que la fonction finale ou la canonique
        if ':' in dep.label:
            if self.deep_and_surf == 'surf':
                dep.label = dep.label[:dep.label.find(':')]
            else:
                i = dep.label.find(':')
                final_label = dep.label[:i]
                can_label = dep.label[i+1:]

                dep.label = can_label
                if final_label != can_label:
                    dep.final_label = final_label

        # if both or deep: on garde la trace du statut de la dep
        if self.deep_and_surf in ['both','deep']:
            dep.deep = deep
            dep.surf = surf
            
        self.deps.append(dep)
        if gov_lidx in self.governor2deps:
            self.governor2deps[gov_lidx].append(dep)
        else:
            self.governor2deps[gov_lidx] = [dep]
        if dep_lidx in self.dependent2deps: 
            self.dependent2deps[dep_lidx].append(dep)
        else:
            self.dependent2deps[dep_lidx] = [dep]
        return True

    def add_dep_from_lidx(self, gov_lidx, dep_lidx, label='unk', pgovlidx=None, plabel=None):
        """ Adds a dependency between two nodes identified by their lidx
        The nodes must be in the graph already
        """
        gov = self.get_lexnode(gov_lidx)
        dep = self.get_lexnode(dep_lidx)
        if gov != None and dep != None:
            self.add_dep( Dep(gov, dep, label, pgovlidx, plabel) )

# repris de Enrique (do_passage_conv)
    def remove_dep(self, dep):
        if self.deps.count(dep) < 1:
            return

        dptdeplist = self.dependent2deps[dep.dependent.lidx]
        govdeplist = self.governor2deps[dep.governor.lidx]
        dptdeplist.remove(dep)
        govdeplist.remove(dep)
        if len(dptdeplist) == 0:
            del self.dependent2deps[dep.dependent.lidx]
        else:
            self.dependent2deps[dep.dependent.lidx] = dptdeplist
        if len(govdeplist) == 0:
            del self.governor2deps[dep.governor.lidx]
        else:
            self.governor2deps[dep.governor.lidx] = govdeplist
        self.deps.remove(dep)
        
    
    def interpret_functional_paths(self, sep='::'):
        """ try to interpret labels of the form mod::dep.suj as : 
        surface label = mod, 
        deep label = dep, 
        deep head = the suj of the surface head """
        for dep in self.deps:
            i = dep.label.find(sep)
            if i == -1:
                continue
            surface_label = dep.label[:i]
            functional_path = dep.label[i+len(sep):]

            # keep trace of the functional path, in features
            dep.dependent.set_feature('fctpath',functional_path)
            # the label holds the surface label
            dep.label = surface_label

            self.follow_functional_path_for_this_dep(dep, functional_path)

    def follow_functional_path_for_this_dep(self, dep, functional_path):

        # get the deep label (first label of functional path)
        path = functional_path.split('/') #('.')
        deep_label = path[0]
        # get the lidx of the deep governor
        path = path[1:]
        path.reverse()
        deep_gov = dep.governor
        for label in path:
            depts = [x for x in self.get_dependents_with_such_labels(deep_gov.lidx, [label]) if x.lidx != dep.dependent.lidx] #if x.cat != 'PROREL']
            l = len(depts)
            if l == 0:
                deep_label = 'NOTFOUND_'+functional_path+str(path)+label+str(deep_gov.lidx)
                deep_gov_lidx = DUMMY_ROOT_LIDX
                break
            if l > 1:
                deep_label += ':ambiguous'
            deep_gov = depts[0]
#pour papier TLT...        print "TRACE:SUBSTITUTE for", dep.dependent.form, dep.dependent.cat, "governor cat", dep.governor.cat, "by", deep_gov.cat
        dep.substitute_surface_by_deep(deep_label, deep_gov)


    def add_gold_information(self, gold_dg, attrname='gglidx'):
        """ Add a feature on each node, that gives the lidx of the governor of that node, as specified in the gold_dg argument """
        for lidx in self.lexnodes:
            node = self.get_lexnode(lidx)
            goldgovnode = gold_dg.get_first_governor(lidx)
            if goldgovnode != None:
                node.set_feature(attrname, goldgovnode.lidx)
            else:
                print("ERROR!!!! could not find a gold governor for node lidx=%d" % lidx)
                print("ERROR!!!! %s"% self.to_string_conll())

    def interpret_fine_tagset(self):
        """ interprets the fine-grained tag into coarsecat + features """
        for lidx in self.lexnodes:
            lexnode = self.lexnodes[lidx]
            if lexnode.cat in FTB4DICT:
                for attr in FTB4DICT[lexnode.cat]:
                    lexnode.set_feature(attr, FTB4DICT[lexnode.cat][attr])
            # for tags ending with + (expanded compounds... shared task SPMRL 2013...)
            elif lexnode.cat.endswith('+') and lexnode.cat[:-1] in FTB4DICT:
                for attr in FTB4DICT[lexnode.cat[:-1]]:
                    lexnode.set_feature(attr, FTB4DICT[lexnode.cat[:-1]][attr])

    def is_lidx_an_auxiliary(self, lidx):
        deps = [x for x in self.get_dep_by_dependent(lidx) if x.label in AuxLabels ]
        if deps:
            return True
        return False

    def add_verbal_information(self):
        """ for verbal predicates (any verb that does not depend on a gov with an auxiliary dependency label)
         => skips auxiliaries, 
         - marks voice and deep mood, 
         - computes "full" lemma (reflexive, causative)
         """
        for lidx in self.lexnodes:
            lexnode = self.lexnodes[lidx]
            if lexnode.coarsecat != 'V' or lexnode.cat.endswith('+'):
                continue
            cat = lexnode.cat
            lidx = lexnode.lidx

            # skip auxiliaries
            if self.is_lidx_an_auxiliary(lidx):
                continue

            voice = 'act'
            # recuperation du "mode profond" (celle du premier V)
            # et de la voie
            mood = FTB4DICT[cat]['m']
            if cat == 'VPP':
                # eventuels auxilaires, triés pour récuperer le 1er
                auxs = sorted(self.get_dependents_with_such_labels(lidx, AuxLabels),
                              lambda x,y: cmp( x.lidx, y.lidx))
                if auxs:
                    auxcat = auxs[0].cat
                    if auxcat in FTB4DICT and 'm' in FTB4DICT[auxcat]:
                        mood = FTB4DICT[auxcat]['m']
                    # bug possible A taggé P, mais predit aux.tps par la syntaxe...
                    elif auxcat == 'P':
                        mood = 'ind'
                    else:
                        mood = ''
                    auxpass = self.get_dependents_with_such_labels(lidx, ['aux.pass'])
                    # si aux du passif ET pas de reflechi (cf. parfois la prediction aux.pass est fausse, impossible si reflechi)
                    if auxpass and not(self.has_refl(lidx)):
                        voice = 'pass'
                # si pas d'aux (participiale)
                # et non conjugué avec etre => passif
                elif lexnode.form not in GramForms['etre_pp']:
                    voice = 'pass'

            lexnode.set_feature('voice',voice)
            lexnode.set_feature('deepmood',mood)
            lexnode.set_feature('fulllemma', self.get_full_lemma(lexnode))


    def sorted_lidx(self):
        """ Returns the sorted list of linear indexes of the nodes that are in the graph, excluding dummy nodes """
        return sorted(self.lexnodes.keys())

    def sorted_dependent_lidx(self):
        """ Returns the sorted list of linear indexes of the nodes that are dependent of something in the graph
        """
        return sorted(self.dependent2deps.keys())

    def get_lexnode(self,lidx):
        """ Returns the node in the graph, having the given lidx 
        Works for the dummy root too"""
        if lidx in self.lexnodes.keys():
            return self.lexnodes[lidx]
        if lidx == DUMMY_ROOT_LIDX:
            return DUMMY_ROOT
        return None

    def get_lidx(self, lexnode):
        if isinstance(lexnode, LexicalNode):
            return lexnode.lidx
        return None

    def get_intervening_nodes( self, lidx1, lidx2 ):
        """ returns the nodes (list) that are linearly between lidx1 and lidx2 """
        if lidx2 < lidx1:
            a = lidx1
            lidx1 = lidx2
            lidx2 = a
        nodes = []
        for i in range(lidx1 + 1, lidx2, 1):
            nodes.append( self.get_lexnode(i) )
        return nodes

    def get_dep_by_dependent(self,dep_lidx,label=None):
        """ Returns the dependencies that have a given dependent
        @param dep_lidx: linear index of dependent lexical node
        @type dep_lidx: integer
        @return: list of Dep (empty list if none exist)"""
        if dep_lidx in self.dependent2deps:
            if not label:
                return self.dependent2deps[dep_lidx]
            return sorted( [dep for dep in self.dependent2deps[dep_lidx] if dep.label == label] )
        return []

    def get_dep_by_governor(self,gov_lidx,label=None):
        """ Returns the dependencies that have a given governor
        @param gov_lidx: linear index of governor lexical node
        @type gov_lidx: integer
        @param label: if set to non-None value, only dependencies with such a label are returned
        @type str
        @return: list of Dep (empty list if none exist)"""
        if gov_lidx in self.governor2deps:
            if not label:
                return self.governor2deps[gov_lidx]
            # tri lineaire
            #return [dep for dep in self.governor2deps[gov_lidx] if dep.label == label]
            return sorted( [dep for dep in self.governor2deps[gov_lidx] if dep.label == label],
                           lambda x,y: cmp(x.dependent.lidx, y.dependent.lidx))
        return []

    def get_first_dep_by_dependent(self,dep_lidx):
        """ Returns the first (in random order) dependency that has a given dependent 
        @param dep_lidx: linear index of dependent lexical node
        @type dep_lidx: integer
        @return: first Dep if exists, None otherwise"""
        a = self.get_dep_by_dependent(dep_lidx)
        if a != []:
            return a[0]
        return None


    def get_governors(self,dep_lidx):
        """ Returns the governors of a given dependent
        (for dependency trees : at most one gov)
        @param dep_lidx: linear index of dependent lexical node
        @type dep_lidx: integer
        @return: List of governors
        @rtype: list of LexicalNode
        """
        return map(lambda x: x.governor, self.get_dep_by_dependent(dep_lidx))

    def get_first_governor(self,dep_lidx):
        """ Returns the first governor of a given dependent
        @param dep_lidx: linear index of dependent lexical node
        @type dep_lidx: integer
        @return: first governor, if it exists, None otherwise
        @rtype: LexicalNode
        """
        dep = self.get_first_dep_by_dependent(dep_lidx)
        if dep != None:
            return dep.governor
        return None


    def get_dependents(self,gov_lidx):
        """ Returns the dependents of a given governor
        @param gov_lidx: linear index of dependent lexical node
        @type gov_lidx: integer
        @rtype: list of LexicalNode (maybe empty)
        """
        return map(lambda x: x.dependent, self.get_dep_by_governor(gov_lidx))

    # new for FeatureExtractionFromLexicalNodes
    def get_ith_dependent_lidx(self, gov_lidx, i, other_than_lidxs=None):
        """ Returns the lidx of the ith dependent in linear order
        if i > 0 : returns the ith dependent of gov_lidx appearing to the right of gov_lidx
        if i < 0 : returns the -ith dependent of gov_lidx appearing to the left of gov_lidx

        if other_than_lidxs is set to a list of lidxs: those are ignored
        """
        a = [ x.lidx for x in self.get_dependents(gov_lidx) ]
        # remove other_than_lidxs from dependent list
        if other_than_lidxs:
            a = filter(lambda x: x not in other_than_lidxs, a)

        if i == 0:
            return None
        if i > 0:
            dependents = sorted( filter( lambda x: x > gov_lidx, a ) )
        else:
            dependents = sorted( filter( lambda x: x < gov_lidx, a ) , reverse=True)
            i = -i

        if i <= len(dependents):
            return dependents[i-1]
        # i bigger than arbitrary big num => returns the last dependent
        if i >= 1000:
            return dependents[-1]
        # no such dependent at the right/left of gov_lidx
        return None 


    def get_dependents_with_such_labels(self,gov_lidx,labels):
        """ Returns the dependents of a given governor, with label in labels
        @param gov_lidx: linear index of dependent lexical node
        @type gov_lidx: integer
        @param labels: list of dependency labels (strings)
        @rtype: list of LexicalNode (maybe empty)
        """
        return [ x.dependent for x in self.get_dep_by_governor(gov_lidx) if x.label in labels ]

    def get_first_dependent_with_such_labels(self,gov_lidx,labels, other_than_lidxs=None):
        """ Returns the first dependent of a given governor, with label in labels
        (first = randomly first!)
        or None if it does not exist
        if other_than_lidxs is specified, as a list of lidxs,
        tries to return the first dependent that is not contained in that list
        """
        dpts = self.get_dependents_with_such_labels(gov_lidx, labels)
        if dpts:
            if not other_than_lidxs:
                return dpts[0]
            else:
                for d in dpts:
                    if d.lidx not in other_than_lidxs:
                        return d
        return None

    def get_dependencies(self):
        """ Returns the dependencies of the DepGraph
        @rtype: list of Dep
        """
        return self.deps

    def sorted_dependencies(self):
        """ Returns the dependencies, sorted by the default sort for Dep: increasing lidx of dependent """
        return sorted(self.deps)

    def get_such_dependency(self, gov_lidx, dep_lidx):
        """ Returns the dependency between governor lidx and dependent lidx if it exists. Returns None otherwise"""
        for dep in self.get_dep_by_dependent(dep_lidx):
            if dep.governor.lidx == gov_lidx:
                return dep
        return None

    def get_dummy_root_lidx(self):
        return DUMMY_ROOT_LIDX

    def get_roots(self):
        """ The roots are the nodes without governor 
        or that only have DUMMY_ROOT as governor
        @return : List of LexicalNodes """
        return [x for x in self.lexnodes.values() if x.lidx not in self.dependent2deps or self.get_governors(x.lidx) == [DUMMY_ROOT]]

    def get_first_root(self):
        r = self.get_roots()
        if r != []:
            return r[0]
        return None

    def get_upwards_path(self, lpath, cpath, lidxs, lemmas, terminating_condition, blocking_condition):
        
        currentlidx = lidxs[-1]
        dep = self.get_first_dep_by_dependent(currentlidx)
        if dep == None or dep.governor.is_dummy_root():
            return False
        lpath.append(dep.label)
        cpath.append(dep.governor.cat)
        lemmas.append(dep.governor.lemma)
        lidxs.append(dep.governor.lidx)
        if eval(terminating_condition):
            return True
        if eval(blocking_condition):
            return False
        return self.get_upwards_path(lpath, cpath, lidxs, lemmas, terminating_condition, blocking_condition)

    
    def get_reachable_nodes(self, lidx):
        """ Computes the nodes reachable from a node
        @param lidx : linear id of node
        @return reachables : recursively constructed set of reachable nodes (lidx of nodes)
        @rtype reachables : set of lidx

        @precondition : the graph is acyclic ...
        """ 
        reachables = set([])
        self.___get_reachable_nodes(lidx, reachables)
        return reachables
    

    def ___get_reachable_nodes(self, lidx, reachables):
        """ Recursive internal function to compute reachable nodes
        @param reachables : recursively constructed set of reachable nodes (lidx of nodes)
        @type reachables : set of lidx
        """
        for node in self.get_dependents(lidx):
            reachables.add(node.lidx)
            self.___get_reachable_nodes(node.lidx, reachables)
            
    def get_node_projection(self,lidx):
        """ Returns the projection of a node : the sequence of lidx dominated by the node """
        return sorted([lidx] + [x.lidx for x in self.get_reachable_nodes(lidx)])

    def compute_icat(self, node):
        """ Returns the category of node, in tagset FTBI,
        with special handling of past participles : 
        they get the category of their first left-most auxiliary, if it exists
        """
        cat = node.cat
        icat = TagSetMap_ftb4_ftbi[cat] if cat in TagSetMap_ftb4_ftbi else cat
        if cat == 'VPP':
            lidx = 1000
            auxs = self.get_dependents_with_such_labels(node.lidx, ['aux.tps', 'aux.pass', 'aux.caus'])
            for aux in auxs:
                if aux.lidx > lidx: break
                icat = TagSetMap_ftb4_ftbi[aux.cat]
                lidx = aux.lidx
        return icat

    # Checks

    def is_empty(self):
        return (len(self.lexnodes.keys()) <= 0)

    def has_dependents(self,gov_lidx):
        return len(self.get_dependents(gov_lidx)) > 0

    def has_det(self, gov_lidx):
        """ Returns true if such lidx has at least one dependent with cat=determiner, or with dependency label 'det' """
        for dep in self.get_dep_by_governor(gov_lidx):
            if dep.dependent.cat in ['DET','D']:
                return True
            if dep.label == 'det':
                return True
        return False

    def has_refl(self, gov_lidx):
        """ if such lidx has at least one dependent with form se, or form=s' and cat=CLR 
        returns the lidx of such dependent 
        returns None otherwise """
        for dep in self.get_dep_by_governor(gov_lidx):
            if dep.dependent.form.lower() == 'se' or (dep.dependent.form.lower() == 's\'' and dep.dependent.cat == 'CLR'):
                return dep.dependent.lidx
        return None

    def has_causative(self, gov_lidx):
        """ if such lidx has a causative auxiliary, returns its lidx
        returns None otherwise """
        for dep in self.get_dep_by_governor(gov_lidx):
            if dep.dependent.get_feature_value('lemma') == 'faire' and dep.label == 'aux.caus':
                return dep.dependent.lidx
        return None

    def get_full_lemma(self,vnode):
        """ builds a lemma including reflexives and causatives
        examples : 'se laver', "s'apercevoir", "faire marcher", "se faire avoir", "faire se laver" 
        """
        # bugs connus : 
        # pb si le réfléchi n'est pas à la 3eme personne!!
        #    nous nous apercevons...
        # pb des autres clitiques figés :" s'en prendre"
        lidxs = {}
        lidxs['refl'] = self.has_refl(vnode.lidx)
        lidxs['caus'] = self.has_causative(vnode.lidx)
        fulllemma = vnode.feature_value_to_string('lemma')
        if (lidxs['refl'] != None or lidxs['caus'] != None):
            for l in sorted(lidxs.keys(), lambda x,y: cmp(lidxs[x], lidxs[y])):
                pref = ''
                if l == 'refl' and lidxs['refl']:
                    pref = "s'" if fulllemma[0] in ['a','e','i','o','u','y','é','ê'] else 'se_'
                if l == 'caus' and lidxs['caus']:
                    pref = 'faire_'
                fulllemma = pref + fulllemma
        return fulllemma

    def is_segment_contiguous(self, lidxs):
        """ Returns True if the set of lidxs contains a contiguous sequence of lidxs
        @param lidxs : set of lidxs
        """
        local = sorted(lidxs.copy())
        prev = local.pop(0)
        while local:
            next = local.pop(0)
            if next > prev + 1:
                return False
            prev = next
        return True
        
    def is_segment_contiguous_ignore(self, lidxs, ignorecats):
        """ Returns True if the set of lidxs either contains a contiguous sequence of lidxs, 
            returns "i" if the segment contains only gaps whose corresponding nodes have a category listed in ignorecats
            returns False otherwise
        Used to determine whether node is projective, if some categories are ignored
        @param lidxs : set of lidxs
        """
        local = sorted(lidxs.copy())
        prev = local.pop(0)
        ignored = 0
        while local:
            next = local.pop(0)
            while (next > prev):
                prev = prev + 1
                if next > prev:
                    if self.get_lexnode(prev).cat not in ignorecats:
                        return False
                    else:
                        ignored += 1
        if ignored > 0:
            return 'i'
        return True
        
    def is_tree(self):
        """ Checks whether the graph is a tree 
        i.e. no node has several governors, and only one node hasn't any governor
        """
        a = [x for x in self.dependent2deps.keys() if len(self.get_governors(x)) > 1]
        return a == [] and len(self.get_roots()) == 1

    def is_projective(self, trace=False):
        """ Checks whether the graph is projective : no node has a projection corresponding to a discontinuous sequence of linear indexes """
        for root in self.get_roots():
            reachables = set([])
            a = self.___is_node_projective(root.lidx, reachables, trace)
            if not a:
                return False
        return True

    def ___is_node_projective(self, lidx, reachables, trace=False):
        """ Recursive internal function to check projectivity of the node """
        #print "___is_node_projective: ", lidx, "==> ", reachables
        # check for all dependents
        for node in self.get_dependents(lidx):
            subreachables = set([])
            if not self.___is_node_projective(node.lidx, subreachables, trace):
                if trace:
                    print("Non projective node :"+str(node.lidx))
                return False
            reachables.update(subreachables)
        # add the starting node itself
        reachables.add(lidx)
        # check for new reachables
        if self.is_segment_contiguous(reachables):
            return True
        if trace:
            print("=>Non projective node :"+str(lidx)+str(self.get_lexnode(lidx).form))
        return False
            
    def would_be_acyclic_and_projective(self, govlidx, deplidx, ignorecats):
        """ tests whether a dependency between govlidx and deplidx would be projective or not, 
        given the current graph minus the dependency between deplidx and its first current gov 
        Returns -2 if the oldgovernor of deplidx is the dummy root : it would be inconsistent to create a depgraph that is not linked to the dummy root
        ortherwise returns -1 if this new dep would create a cycle
        otherwise returns 0 if would not be projective
        otherwise returns 1
        """
        #print "WOULD_BE_ACYCLIC_AND_PROJECTIVE?", govlidx, deplidx, self.get_lexnode(govlidx).form, self.get_lexnode(deplidx).form
        oldgov = self.get_first_governor(deplidx)
        #it would be inconsistent to create a depgraph that is not linked to the dummy root
        if oldgov.is_dummy_root():
            return -2

        # temporarily modify the tree...
        olddep = self.get_such_dependency(oldgov.lidx, deplidx)
        self.remove_dep(olddep)
        newdep = Dep( self.get_lexnode(govlidx), self.get_lexnode(deplidx), 'd' )
        r = self.add_dep(newdep)
        if not r:
            self.add_dep(olddep, check_no_cycle=False)
            return -1

        reachedfromgov = self.get_reachable_nodes(govlidx)
        reachedfromgov.add(govlidx)
        #print "REACHEDFROMGOV", reachedfromgov

        # would the tree from the new governor still be projective?
        r1 = self.is_segment_contiguous_ignore(reachedfromgov, ignorecats) 
        #print "NEW GOV STILL PROJ?", r1

        r2 = False
        # would the tree from the old governor still be projective?
        if r1:
            reachedfromoldgov = self.get_reachable_nodes(oldgov.lidx)
            reachedfromoldgov.add(oldgov.lidx)
            #print "REACHEDFROMOLDGOV", reachedfromoldgov
            r2 = self.is_segment_contiguous_ignore(reachedfromoldgov, ignorecats)
            #print "OLD GOV STILL PROJ?", r2

        # rebuild the original tree
        self.remove_dep(newdep)
        self.add_dep(olddep, check_no_cycle=False)

        if (r1 and r2):
            return 1
        return 0

    def change_governor(self, depnode, newgovnode, label=None, trace=False):
        """ removes the dependency(ies) where depnode is the dependent, and adds a dependency to the given governor, with label supplied
        If no label supplied, takes the label of the first removed dependency"""
        deplidx = depnode.lidx
        newgovlidx = newgovnode.lidx
        removed = False
        for oldgov in  self.get_governors(deplidx):
            if oldgov.is_dummy_root():
                if trace:
                    print("#change_governor of %d (%s) from dummy root to %s" %( deplidx, depnode.form, newgovnode.form ))
                continue
            revision_type = ''
            # compute the "correcteness" of this change, if gold information is available
            goldgovlidx = depnode.get_feature_value('gglidx')
            if goldgovlidx != None:
                old = 'c' if oldgov.lidx ==  goldgovlidx else 'w'
                new = 'c' if newgovnode.lidx == goldgovlidx else 'w'
                revision_type = old + '2' + new
            if trace:
                print("#change_governor (%s) of %d (%s) from %s to %s [%s]" %(revision_type, deplidx, depnode.form, oldgov.form, newgovnode.form , depnode.cat))

            olddep = self.get_such_dependency(oldgov.lidx, deplidx)
            self.remove_dep(olddep)
            if label == None: label = olddep.label
            removed = True
        # if some dep was actually removed
        if removed:
            self.add_dep( Dep(newgovnode, depnode, label) ) 

    def count_correct_deps(self, ignore_punct=True):
        """ Pertains only if gold information is available on lexical nodes (cf. LexicalNode.add_gold_information)
        Returns the number of dependents that have the correct governor, and the total number of deps
        """
        nb_corrects = 0
        nb_dependents = 0
        for lidx in self.lexnodes:
            node = self.get_lexnode(lidx)
            if ignore_punct and utils.is_punct(node.form):
                continue
            goldgovlidx = node.get_feature_value('gglidx')
            if goldgovlidx != None:
                govnode = self.get_first_governor(lidx)
                if goldgovlidx == govnode.lidx:
                    nb_corrects += 1
                nb_dependents += 1
        return (nb_corrects, nb_dependents)
                
    def get_shortest_paths(self, lidx1, lidx2):
        """ shortest paths (any direction) between nodes lidx1 and lidx2
        Designed to work even in case of cycles
       
        @return: the list of shortest paths between lidx1 and lidx2, 
                 Each path is a list of strings containing the arc label,
                      prefixed by either '+' or '-' depending on the arc orientation
                      [ '+a_obj', 'obj' ]
        NB: because the traversed nodes are not written in the paths, equal paths can be output (but traversing different nodes)
        """
        seen_lidxs = set([lidx1])
        # parcours en etoile à partir de lidx1
        # on stocke les noeuds atteints, avec le chemin correspondant
        # les noeuds atteints sont par construction groupés par longueur du chemin
        # liste de listes: le rang correspond à la longueur du chemin
        #                  valeur à un rang : la liste de couples (lidx, chemin) atteints

        # à la longueur zéro : on a le lidx1 lui-même
        reached_nodes_and_paths = [ [ (lidx1, []) ] ]
        # si lidx2 est lidx2 on a fini
        if lidx2 == lidx1:
            return [ [] ]
        while(1):
            last_layer = reached_nodes_and_paths[-1]
            new_layer = [ ]
            # liste de nouveaux noeuds accessibles a partir de la last_layer
            # NB: un meme noeud peut apparaitre plusieurs fois ds une meme couche, pour plusieurs chemins de longueur min
            new_lidxs = set([])
            for (lidx, path) in last_layer:
                node = self.get_lexnode(lidx)
            # dependants directs
                for dependency in self.get_dep_by_governor(lidx):
                    reached_lidx = dependency.dependent.lidx
                    if reached_lidx in seen_lidxs:
                        continue
                    new_lidxs.add(reached_lidx)
                    new_layer.append( (reached_lidx, path + [ '+'+dependency.label]) )
            # gouverneurs directs (sortir le dummy)
                for dependency in self.get_dep_by_dependent(lidx):
                    reached_lidx = dependency.governor.lidx
                    if reached_lidx == DUMMY_ROOT_LIDX or reached_lidx in seen_lidxs:
                        continue
                    new_lidxs.add(reached_lidx)
                    new_layer.append( (reached_lidx, path + [ '-'+dependency.label]) )
            #si aucun nouveau noeud, on n'a pas reussi à atteindre lidx2 => on sort
            if len(new_lidxs) == 0:
                return []
            # si lidx2 est atteint, on rend la liste des plus courts chemins pour l'atteindre
            paths_to_lidx2 = [ x[1] for x in new_layer if x[0] == lidx2 ]
            if len(paths_to_lidx2) > 0:
                return paths_to_lidx2

            # sinon on stocke la nouvelle layer, et les nouveaux noeuds
            reached_nodes_and_paths.append(new_layer)
            seen_lidxs = seen_lidxs.union(new_lidxs)



            
    # Printing functions

    def __str__(self):
        return self.to_string_pivot()

    def get_yield_str(self, attr='form',separator=' ',normalize=True):
        """ Returns a string made of the nodes form, separated by separator """
        return separator.join([self.get_lexnode(x).feature_value_to_string(attr,normalize=True) for x in self.sorted_dependent_lidx()])

    def to_string_pivot(self, features=['cat']):
        lidxs = self.sorted_dependent_lidx() # consider only nodes that are dependent of something in the graph
        res = 'sentence_form('+self.get_yield_str(normalize=True)+')\n'
        res = res + 'surf_deps(\n'
        res = res + '\n'.join([ self.get_first_dep_by_dependent(lidx).to_string_pivot() for lidx in lidxs ])+'\n'
        res = res + ')\n' # close surf_deps
        res = res + 'features(\n'
        if features != None:
            for attr in features:
                res = res + '\n'.join([ self.get_lexnode(lidx=lidx).feature_to_string_pivot(attr) for lidx in lidxs ])+'\n'
        res = res + ')\n' # close features
        return res

    def to_string_conll(self):
        s = '\n'.join(map(lambda x: self.get_first_dep_by_dependent(x).to_string_conll(), self.sorted_dependent_lidx()))
        if s:
            return s+'\n'
        return s

    def to_string_deep_conll(self):
        """ returns a deep conll string : multiple governors of a dependent written as a pipe-separated list """
        s = ''
        for deplidx in self.sorted_lidx():
            dependent = self.get_lexnode(deplidx)
            s += dependent.to_string_conll()            
            govs = []
            labels = []
            plabel = '_'
            pgov = '_'
            for dep in self.get_dep_by_dependent(deplidx):
                govs.append(str(dep.governor.lidx + 1))
                labels.append(dep.label)
                # on ne gere qu'un seul plabel/pgov...
                pgovlidx=dep.get_feature_value('pgovlidx')
                if pgovlidx:
                    plabel = dep.feature_value_to_string('plabel','_') 
                    pgov = str(pgovlidx + 1)
            s += '\t'+'|'.join(govs)+'\t'+'|'.join(labels)+'\t'+pgov+'\t'+plabel+'\n'

        return s+'\n'

    def get_sentid_from_first_node(self):
        #print("SENTID: %s" % str(self.get_lexnode(0).get_feature_value('sentid')))
        return self.get_lexnode(0).get_feature_value('sentid')
    


class DepParse:
    def __init__(self, sentid, depgraph, parseid=1, status=None,features=None, surf_dg=None):
        """ Structure to store a depgraph, along with its meta information :
        @param sentid : identifier of the sentence : either a string identifier from a treebank, or a rank if comes from a parsed file; CAUTION if depgraph has a sentid encoded on its first lexical node, it *overrides* the sentid given as argument
        @type sentid : string

        @param parseid : parse rank for this sentence (default=1)
        @type parseid : integer

        @param status : (TODO) status of the parse : gold, autofromfunc, autofrommrg, autofromraw ... default=None
        @type : either None or string
        
        @param depgraph : the dependency graph
        @type depgraph : DepGraph

        @param features : Additional fields 
        @type features : either None, or dict mapping field / value

        @surf_dg : additional depgraph, if depgraph is deeponly
        @type surf_dg : DepGraph
        """
        # the dependency graph
        self.depgraph = depgraph

        # meta information
        if isinstance(sentid, str) or sentid == None:
            self.sentid = sentid
        else:
            self.sentid = str(sentid)
        # take the id defined on first lexical node, if defined
        self.set_id_from_depgraph()

        self.parseid = parseid
        self.status = status

        self.surf_dg = surf_dg
        # additional features
        if features != None:
            self.__dict__.update(features)



    def get_feature_value(self,attr):
        """ Returns the value of the given attribute, if defined,
        otherwise returns None 
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        return None

    # Printing functions 
    def __str__(self):
        return self.to_string_pivot()

    def feature_value_to_string(self,attr,dummy=None,normalize=False):
        a = self.get_feature_value(attr)
        if dummy != None and a == None:
            return dummy
        # if normalize:
        #     if isinstance(a, str):
        #         return encode_const_metasymbols(a)
        #     return encode_const_metasymbols(str(a))
        return str(a)


    def to_string_pivot(self, features=['cat']):
        """ returns a string for the parse (meta-information + depgraph) in 'pivot' format """
        res = 'sentence(\n'
        for attr in ['sentid','date','validators']:
            if attr in self.__dict__:
                res = res + attr + '(' + self.feature_value_to_string(attr,normalize=True) + ')\n'
        res = res + self.depgraph.to_string_pivot(features)
        return res + ')\n' # close sentence

    def add_id_to_depgraph(self):
        dg = self.depgraph
        firstlex = dg.get_lexnode(0)
        firstlex.set_feature('sentid',self.sentid)

    def set_id_from_depgraph(self):
        dg = self.depgraph
        firstlex = dg.get_lexnode(0)
        # on utilise en priorite le trait sentrk (cf. pb salto : ids numeriques)
        ##!! non pas ici
        ## sentid = firstlex.get_feature_value('sentrk')
        sentid = firstlex.get_feature_value('sentid')
        if not sentid:
            sentid = firstlex.get_feature_value('sentid')
        if sentid != None:
            self.sentid = sentid
            dg.sentid = sentid
