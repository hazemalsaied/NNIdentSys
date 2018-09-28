import logging

from xpTools import *

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(level=logging.WARNING)
    setXPMode(XpMode.compo)
    setDataSet(Dataset.sharedtask2)

    setTrainAndTest(Evaluation.fixedSize)
    xp(['EN'], xpNum=1)
    # xp(allSharedtask2Lang, xpNum=1)
