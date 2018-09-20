import logging

from config import *
from corpus import *
from identification import xp, setXPMode, setTrainAndTest, setDataSet
from xpNonCompo import allSharedtask2Lang

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    setXPMode(XpMode.compo)
    setDataSet(Dataset.sharedtask2)

    setTrainAndTest(Evaluation.fixedSize)
    xp(allSharedtask2Lang, xpNum=1)
