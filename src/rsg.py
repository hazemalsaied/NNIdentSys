import math
import pickle
import random
import sys

fileName = 'rsgGrid.p'
configuration = dict()


def createRSGrid(xpNum=500):
    xps = dict()
    for i in range(xpNum):
        # int parameter
        configuration['parameter1'] = int(generateValue([50, 500], True, True))
        # Float parameter
        configuration['parameter2'] = float(generateValue([10, 100], True, True))
        # ...
        # Boolean parameter
        configuration['parameterN'] = generateValue([True, False], False, False)

        xps[i] = [False, dict(configuration)]
    pickle.dump(xps, open(fileName, 'wb'))
    return xps


def runRSG(xpNumByThread=60):
    for i in range(xpNumByThread):
        # reload the grid for every iteration
        exps = pickle.load(open(fileName, 'rb'))
        while True:
            xpIdx = random.randint(1, len(exps)) - 1
            exp = exps.keys()[xpIdx]
            if not exps[exp][0]:
                break
        exps[exp][0] = True
        pickle.dump(exps, open(fileName, 'wb'))
        # Assign the new onfiguration
        configuration.update(exps[exp][1])
        xp()


def generateValue(plage, continousPlage=False, uniform=False, favorisationTaux=0.6):
    if continousPlage:
        if uniform:
            # Uniform distribution
            return random.uniform(plage[0], plage[-1])
        else:
            # Geometrically drawn
            return pow(2, random.uniform(math.log(plage[0], 2), math.log(plage[-1], 2)))
    else:
        return plage[random.randint(0, len(plage) - 1)] if uniform else \
            plage[0] if random.uniform(0, 1) < favorisationTaux else plage[random.randint(1, len(plage) - 1)]


def xp():
    # ton experience
    pass


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    runRSG()
