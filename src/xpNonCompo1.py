from config import *
import sys
from xpNonCompo import exploreEmbImpact, dautresXps2

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    samling = configuration["sampling"]
    samling["importantSentences"] = True
    samling["overSampling"] = True
    samling["sampleWeight"] = True
    samling["favorisationCoeff"] = 10

    dautresXps2()
    # exploreEmbImpact([40], posEmbs=[25, 35, 50], denseDomain=[25, 75], useLemma=True, usePos=True)
