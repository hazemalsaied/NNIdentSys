import asOracle
from corpus import *


def parse(lang):
    corpus = Corpus(lang)
    for i in corpus.depLblMgr.depLbls:
        print i
    asOracle.parse(corpus)


if __name__ == '__main__':
    parse('FR')
