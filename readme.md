ATILF-LFF is a generic transition system for the identification of Verbal Multiword Expressions (VMWEs). This system is a data-driven system applicable to several languages. It produces a robust, efficient performance and very competitive scores, with respect to the shared task results.

The system was developed and evaluated using the datasets of the PARSEME shared task on VMWE identification and accommodates the variety of linguistic resources provided for each language, in terms of accompanying morphological and syntactic information. 

##Description en français:

ATILF-LFF est un système d'identification des EPs (verbales) à base de transitions . Ce système est indépendant de la langues et guidée par les données. Il présente une performance robuste et efficace et ses scores incarnent l'état de l'art du domaine de l'identification des EPs. 
Ce système a été mis en oeuvre et évalué grâce aux données de la campagne internationale de l'identification des expressions polylexicales verbales et nous avons essayé d'exploiter le maximum des ressources linguistiques fournies pour chaque langue. Des modèle linéaires et neuronaux ont été construits et y intégrés et un ensemble spécial de transitions a été proposé pour améliorer sa puissance expressive. 

## Citation:
    Al Saied, Hazem, Matthieu Constant, and Marie Candito. 
    "The atilf-llf system for parseme shared task: a transition-based verbal multiword expression tagger."
    Proceedings of the 13th Workshop on Multiword Expressions (MWE 2017). 2017.

## Libraries:

    conda install numpy=1.13.1
    conda install sklearn=0.18.1
    conda install keras
    conda install theano
    conda install matplotlib
    conda install -c conda-forge imbalanced-learn
    conda install pytorch torchvision -c pytorch
    conda install -c anaconda word2vec
    conda install -c anaconda gensim
    conda install -c anaconda statsmodels

## Ressources:

### Sharedtask 1.0:

    https://goo.gl/4agiDo

### Sharedtask 1.1:

    https://goo.gl/LNatQv

### French TreeBank FTB:

    http://ftb.linguist.univ-paris-diderot.fr/telecharger.php

### DiMSUM:

    https://github.com/dimsum16/dimsum-data