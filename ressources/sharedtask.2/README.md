README
------
Welcome to the data repository of the [PARSEME shared task on automatic identification of verbal multiword expressions - edition 1.1](http://multiword.sourceforge.net/sharedtask2018).

This repository contains corpora in multiple languages.
Corpora were annotated by human annotators with occurrences of verbal multiword 
expressions (VMWEs) according to common [annotation guidelines](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/).

You can download this repository as a [ZIP file](https://gitlab.com/parseme/sharedtask-data/repository/archive.zip?ref=master).


Corpora
-------
Annotations can be found for multiple language directories,
such as `FR` for French, `DE` for German, `PL` for Polish, etc.
Inside each language directory, you may find these files:

  * `README.md`: A description of the available data for that given language.
  * `train.cupt`: Training data in [cupt format](http://multiword.sourceforge.net/cupt-format).
  * `dev.cupt`: Development data in [cupt format](http://multiword.sourceforge.net/cupt-format).
  * `test.blind.cupt`: The blind test data (released to participants on April 30, 2018)
  * `test.cupt`: The gold test data (released to participants after the shared task evaluation phase, on May 11, 2018)
  * `{train,test,dev}-stats.md`: Number of sentences, tokens and annotated VMWEs in each part of the corpus.

The *cupt* files are an extension of the [CoNLL-U](http://universaldependencies.org/format.html)
format containing the original 10 columns from CoNLL-U + 1 additional column 
containing VMWE annotations. Depending on the language, different types of 
information are available in the first 10 columns (check the language-specific 
`README.md` files).

Note: For some languages, some fields may contain data that does not use the 
Universal Dependencies tagsets. We thought it would be useful to provide this 
information nevertheless.


Scripts
-------
The `bin` directory contains useful scripts:

  * `bin/evaluate.py`: script that assesses system predictions according to gold annotations by calculating the [evaluation metrics](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018&subpage=CONF_50_Evaluation_metrics).
  * `bin/validate_cupt.py`: script for checking if the prediction file is in the proper format.
  * `bin/parsemetsv2cupt.py`: script that converts the old PARSEME-TSV format of shared task 1.0 (2017) into the new cupt format.
  * `bin/average_of_evaluations.py`: script used to calculate cross-lingual macro-averages for general and phenomenon-specific metrics.


Trial data
----------

The `trial` directory contains trial data:
  * `trial/EN_trial-test_pred.cupt`: Example of file containing predicted VMWEs in English.
  * `trial/EN_trial-test_gold.cupt`: Example of file containing gold VMWE annotations in English.
  * `trial/EN_trial-train.cupt`: Example of file containing a training corpus in English.

