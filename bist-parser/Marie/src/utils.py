#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import re
import unicodedata
import sys
from random import sample

def split_list(inlist, proportion=0.1, shuffle=True):
     """ randomly partitions the input list of items (of any kind) into 2 lists, the first one representing @proportion of the whole """
     n = len(inlist)
     size1 = int(n * proportion)
     if not(size1):
          size1 = 1
     sys.stderr.write("SPLIT %d sentences into %d and %d\n" % (n, n-size1, size1))
     # if shuffle : simply shuffle and return slices
     if shuffle:
          # shuffle inlist (without changing the original external list
          # use of random.sample instead of random.shuffle
          inlist = sample(inlist, n)
          return (inlist[:size1], inlist[size1:])
     # otherwise, return validation set as one out of xxx item
     else:
          divisor = int(n / size1)
          l1 = []
          l2 = []
          for (i,x) in enumerate(inlist):
               if i % divisor or len(l1) >= size1:
                    l2.append(x)
               else:
                    l1.append(x)
          return (l1,l2)
     
def load_mapping(mapping_file, sep='\t', key_col=0, val_col=1, threshold_col=-1, minocc=0, val2_col=None, val_type=None):
     """ Loads mapping_file, splits line using sep, and maps the key to the value, the key is taken at rank key_col, the value is taken at rank val_col (ranks start at 0) 
     The last value for a given key is retained in case of multiple values for one key.
     @return dictionary containing the mapping
     """
     mapping = {}
     instream = open(mapping_file)
     line = instream.readline()
     while line:
         line = line[0:-1]
         cols = line.split(sep)
         if len(cols) >= max(key_col, val_col):
              # skip if a threshold is specified, and is not reached
              if threshold_col > -1 and minocc > 0 and len(cols) >= threshold_col and int(cols[threshold_col]) < minocc:
                   line = instream.readline()
                   continue
              # !!!!hack tempo!!!! pour charger les clusters avec forme de type  "au_-_dessous_de" et pas "au-dessous_de"
              # !!!! il faudra changer le tokeniseur pour produire les "au_-_dessous_de" (traitement du tiret)
              key2 = None
              if cols[key_col].find('-') > -1:
                   key2 = cols[key_col].replace('-','_-_')
              if val_type != None:
                   cols[val_col] = eval(val_type)( cols[val_col] )
              if val2_col != None and val2_col < len(cols):
                   if val_type != None:
                        cols[val2_col] = eval(val_type)( cols[val2_col] )
                   mapping[cols[key_col]] = [ cols[val_col], cols[val2_col] ]
                   if key2 != None:
                        mapping[key2] = [ cols[val_col], cols[val2_col] ]

              else:
                   mapping[cols[key_col]] = cols[val_col]
                   if key2 != None:
                        mapping[key2] = cols[val_col]

         line = instream.readline()
     return mapping

def find_partition_thresholds(list, nb_parts):
     """ given a list of values, and a number of parts, finds the thresholds that divide the list into nb_parts parts, of equal size """
     s_list = sorted(list)
     part_size = float(len(list)) / float(nb_parts)
     int_part_size = int(part_size)
     thresholds = []
     for i in range(nb_parts - 1)  :
          thresholds.append( s_list[ i*int_part_size ] )
     return thresholds

def find_val_partition_thresholds(list, nb_parts):
     """ given a list of values, and a number of parts, finds the thresholds that divide the list into nb_parts parts, so that
         the first part has values between 100% * max and (100 - 100/nb_parts)% *max etc...
         Eg if nb_parts = 10, the first part gives are the values between 100 and 90% of the maximum value
         """
     print("TODO")

def put_into_bin(value, bin_thresholds):
        """ returns the bin corresponding to a given value, given the bin thresholds """
        if value < bin_thresholds[0]:
            return 'b0'
        for (rk, threshold) in enumerate(bin_thresholds[1:]):
            if value < threshold: return 'b'+str(rk+1)
        return 'bmax'


# from Enrique
def is_punct(word):
     """ whether a word is a punctuation, according to unicode categorization (as in eval07 conll script """
     uword = unicode(word, 'iso-8859-1')
     for uchar in uword:
          if unicodedata.category(uchar)[0] != 'P':
               return False
     return True

class TagFixer:
      """
      A TagFixer is a class mapping a conflated symbol like VINF to a base category V and (optionally) a feature structure
      e.g. VINF-> basecat=V, mood=inf, NC --> basecat=N, subcat=common
      """
      def __init__(self):
            """
            Creates an empty tag fixer with a type definition specified by attlist
            """
            self.mappings = {}

      def set_tagmap(self,orig_tag,dest_tag,list_of_features=[]):
            """
            This sets a mapping from a conflated tag to its analytical decomposition.
            e.g. 'VINF' --> 'V', ('m','inf')
            
            @param orig_tag: the tag to analyse
            @type orig_tag: a string
            @param dest_tag: a base category
            @type dest_tag: a string
            @param list_of_features: an optional list of couples (2-tuples) of the form (attribute,value)
            @type list_of_features: a list of 2-tuples of strings
            """
            self.mappings[orig_tag] = (dest_tag,list_of_features)
		
      def map_tag(self,tag):
            """
            This returns a couple of the form (basecat,list_of_features) being the analytical decomposition of a tag.
            
            @param tag: the tag from which to derive the analytical form
            @type tag: string
            @return:a couple (basecat,list_of_features) where list of features is None if there are no features. In case there is no mapping for tag, returns the tag itself with an empty list of features
            @rtype: a couple(string,list of couples of strings)
            """
            if self.mappings.has_key(tag):
                  return self.mappings[tag]
            elif tag.endswith('+') and self.mappings.has_key(tag[:-1]):
                  return self.mappings[tag[:-1]]
            else:
                  return (tag,[])

      # marie : construction du mapping inverse, pour la conversion basecat + features => complexcat
      def build_inverse_mapping(self):
            """ Builds the inverse mapping : from basecat to complexcat + features
            in order to facilitate the computation of complexcat from basecat + features
            The method sets the dictionary inverse_mappings
            key = basecat, 
            value = list of pairs : (complexcat, dictionary of features)
                    the list of pairs is sorted, with decreasing order of number of features (so that (ADJINT, {subcat:1}) appears before (ADJ, {}))
                    """
            self.inverse_mappings = {}
            for complextag in self.mappings:
                  (basecat, list_of_features) = self.map_tag(complextag)
                  dfeats = dict(list_of_features)
                  if basecat in self.inverse_mappings:
                        self.inverse_mappings[basecat].append( (complextag, dfeats) )
                        # sort the feature dictionaries with decreasing order of keys
                        s = sorted(self.inverse_mappings[basecat], 
                                   lambda x,y: cmp(len(y[1].keys()),len(x[1].keys())))
                        self.inverse_mappings[basecat] = s
                  else:
                        self.inverse_mappings[basecat] = [ (complextag, dfeats) ]
            
def ftb4_fixer():

     tfixer = TagFixer()
     tfixer.set_tagmap('V','V',[('m','ind')])
     tfixer.set_tagmap('VINF','V',[('m','inf')])
     tfixer.set_tagmap('VIMP','V',[('m','imp')])
     tfixer.set_tagmap('VS','V',[('m','subj')])
     tfixer.set_tagmap('VPP','V',[('m','part'),('t','past')])
     tfixer.set_tagmap('VPR','V',[('m','part'),('t','pst')])
     # marie à valider : pb cf. NC est la cat par défaut pour subcat différente de p (subcat = C ou subcat = card)
     # => introduire disjonction sur les traits, avec une valeur prioritaire
     #tfixer.set_tagmap('NC','N',[('s','c')])
     
     tfixer.set_tagmap('NC','N', [])
     tfixer.set_tagmap('NPP','N',[('s','p')])
     
     # marie debug (?) for the subcat attr, keep the values as in original FTB
     #tfixer.set_tagmap('CS','C',[('s','sub')])
     #tfixer.set_tagmap('CC','C',[('s','coord')])
     tfixer.set_tagmap('CS','C',[('s','s')])
     tfixer.set_tagmap('CC','C',[('s','c')])
     
     # marie debug (?) for the subcat attr, keep the values as in original FTB
     #tfixer.set_tagmap('CLS','CL',[('s','subj')])
     tfixer.set_tagmap('CLS','CL',[('s','suj')])
     tfixer.set_tagmap('CLO','CL',[('s','obj')])
     tfixer.set_tagmap('CLR','CL',[('s','refl')])
     tfixer.set_tagmap('CL','CL',[('s','')])
     
     tfixer.set_tagmap('ADJWH','A',[('s','int')])
     tfixer.set_tagmap('ADJ','A',[])
     
     tfixer.set_tagmap('ADVWH','ADV',[('s','int')])
     tfixer.set_tagmap('ADV','ADV',[])
     
     tfixer.set_tagmap('PROREL','PRO',[('s','rel')])
     tfixer.set_tagmap('PROWH','PRO',[('s','int')])
     tfixer.set_tagmap('PRO','PRO',[])
     
     tfixer.set_tagmap('DET','D',[])
     tfixer.set_tagmap('DETWH','D',[('s','int')])
     
     tfixer.set_tagmap('P+D','P+D',[('s','def')])
     tfixer.set_tagmap('P+PRO','P+PRO',[('s','rel')])
     return tfixer

# mapping from fine-grained categories (ftb4) to coarse cat
TagSetMap_ftb4_ftbmin = {
     'V':'V',
    'VINF':'V',
    'VIMP':'V',
    'VS':'V',
    'VPP':'V',
    'VPR':'V',
    'NC':'N',
    'NPP':'N',
    'CS':'C',
    'CC':'C',
    'CLS':'CL',
    'CLO':'CL',
    'CLR':'CL',
    'CL':'CL',
    'ADJ':'A',
    'ADJWH':'A',
    'ADV':'ADV',
    'ADVWH':'ADV',
    'PRO':'PRO',
    'PROREL':'PRO',
    'PROWH':'PRO',
    'DET':'D',
    'DETWH':'D',
     'P':'P',
     'P+D':'P+D',
     'P+PRO':'P+PRO',
     'PONCT':'PONCT',
     'ET':'ET',
     'I':'I'
    }

# mapping from fine-grained categories
# to a coarser set, yet finer than the coarse FTB tagset
TagSetMap_ftb4_ftbi = {
    'V':'V',
    'VINF':'VINF', #!
    'VIMP':'V',
    'VS':'V',
    'VPP':'VPP', #!
    'VPR':'VPR', #!
    'NC':'N',
    'NPP':'N',
    'CS':'CS', #!
    'CC':'CC', #!
    'CLS':'CL', 
    'CLO':'CL',
    'CLR':'CL',
    'CL':'CL',
    'ADJ':'A',
    'ADJWH':'A',
    'ADV':'ADV',
    'ADVWH':'ADV',
    'PRO':'PRO',
    'PROREL':'PROREL', #!
    'PROWH':'PRO',
    'DET':'D',
    'DETWH':'D',
    'P':'P',
    'P+D':'P', #!
    'ET':'ET',
    'I':'I',
    'PONCT':'PONCT'
    }

# utilisé pour deviner la cat fine des composants de composés
Special_tokens_2_ftb4_cat = {
     ('tandis','C'):'CS',
     ('tant','C'):'CS',
     ('que','C'):'CS',
     ('qu\'','C'):'CS',
     ('parce','C'):'CS',
     ('y','CL'):'CLO',
     ('en','CL'):'CLO',
     ('se','CL'):'CLR',
     ('s\'','CL'):'CLR',
     ('vous','CL'):'CLO', # vérifié
     ('il','CL'):'CLS',
     ('elle','CL'):'CLS',
     ('on','CL'):'CLS',
     ('eux','CL'):'PRO', # correction erreur!
     ('après','C'):'P', # correction erreur!
     ('ce','CL'):'CLS',
     ('c\'','CL'):'CLS',
     ('si','C'):'CS',
     ('s\'','C'):'CS',
     ('et','C'):'CC',
     ('ni','C'):'CC',
     ('même','C'):'CC',
     ('mais','C'):'CC',
     ('ou','C'):'CC',
     ('&','C'):'CC',
     ('soit','V'):'VS',
     ('faite','V'):'VPP',
     ('faites','V'):'VPP',
     ('dit','V'):'VPP', # pb car souvent VPP, mais parfois V
     # suffixes...
     ('u','V'):'VPP',
     ('ue','V'):'VPP',
     ('us','V'):'VPP',
     ('ues','V'):'VPP',
     ('i','V'):'VPP',
     ('ie','V'):'VPP',
     ('is','V'):'VPP',
     ('ise','V'):'VPP',
     ('ses','V'):'VPP', # j'ai vérifié: ce sont bien des VPP
     ('é','V'):'VPP',
     ('és','V'):'VPP',
     ('ée','V'):'VPP',
     ('ées','V'):'VPP',
     ('vre','V'):'VINF',
     ('ir','V'):'VINF',
     ('er','V'):'VINF',
     ('ire','V'):'VINF',
     ('tre','V'):'VINF',
     ('dre','V'):'VINF',
     ('ant','V'):'VPR',
     ('ez','V'):'VIMP',
}

# valeurs FTB4 par défaut, utilisé pour deviner la cat fine des composants de composés
Default_ftbmin2ftb4 = {
     'C':'CS',
     'CL':'CLS',
     'V':'V',
}

# marie: infos de traits plus light
def tagfixer2tagdict(tagfixer):
     dico = {}
     for cat in tagfixer.mappings:
          dico[cat] = dict(tagfixer.mappings[cat][1])
          dico[cat]['coarsecat'] = tagfixer.mappings[cat][0]
     return dico
          

# marie : vite fait : features tels qu'utilisés dans LabelledTree, vers les features du XML ftb
# PREREQUIS : le noeud est un tag node, et il porte une basecat
def treefeats2xmlfeats(node):
     feats = node.features
     mph = ''
     if 's' in feats:
          subcat = feats['s'] 
          if node.basecat in ['N','C','PONCT']:
               subcat = subcat.upper() 
               feats['s'] = subcat
          feats['subcat'] = subcat
          del(feats['s'])
     else:
          subcat = ''
     if 'm' in feats:
          mood = feats['m']
          tense = ''
          if 't' in feats:
               tense = feats['t']
          if mood == 'inf': mph = 'W'
          elif mood == 'part':
               if tense == 'pst':
                    mph = 'G'
               elif tense == 'past':
                    mph = 'K'
               else:
                    sys.stderr.write('Error: bad tense:'+ str(feats))
          elif mood == 'imp': mph = 'Y'
          elif mood == 'ind':
               if tense == 'pst': mph = 'P'
               elif tense == 'impft': mph = 'I'
               elif tense == 'past': mph = 'J'
               elif tense == 'fut': mph = 'F'
               elif tense == 'cond': mph = 'C'
               else:
                    sys.stderr.write('Warning: missing or wrong tense:'+ str(feats))
                    # cas d'erreur = le hack sur 'il_y_a'
                    mph = 'P3s'
          elif mood == 'subj':
               if tense == 'pst': mph = 'S'
               elif tense == 'impft': mph = 'T'
               else:
                    sys.stderr.write('Warning: missing or wrong tense:'+ str(feats))
          del(feats['m'])
     if 'p' in feats:
          mph += str(feats['p'])
          del(feats['p'])
     if 'g' in feats:
          mph += feats['g']
          del(feats['g'])
     if 'n' in feats:
          mph += feats['n']
          del(feats['n'])

     if 'fct' in feats:
          del(feats['fct'])

     if node.has_function():
          node.fct = node.fct.lower().replace('_','-')

     ei = node.basecat + mph
     if node.basecat == 'V':
          ee = node.basecat + '--' + mph
     elif node.basecat in ['D','N','A','CL']:
          # il manque le nombre du possesseur
          ee = node.basecat + '-' + subcat + '-' + mph
     elif node.basecat in ['C','PONCT']:
          # il manque le nombre du possesseur
          ee = node.basecat + '-' + subcat
     else:
          ee = node.basecat

     feats['ei'] = ei
     feats['ee'] = ee
     feats['mph'] = mph
     node.features = feats
     
               
               
