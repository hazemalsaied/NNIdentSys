# A
useLemma = True
usePOS = True
useToken = True
# A'
useSufixes = False
# B
useSyntax = False
# C
useBiGram = True
# D
useTriGram = False
# E
generateS0B2Bigram = True
# F
historyLength1 = False
# G
historyLength2 = False
# H
historyLength3 = False
# I
useS0B0Distance = True
# J
useS0S1Distance = True
# K
useB1 = True
# L
useLexic = True
enhanceMerge = False
# M
useStackLength = False
# N
smartMWTDetection = False

useAbstractSyntax = True

import json, os


def getConfig(configFolder):
    for filename in os.listdir(configFolder):
        if not filename.endswith(".json"):
            continue
        res = ''
        data = json.load(open(os.path.join(configFolder, filename)))
        if data["UseLinguistInfo"]["usePOS"] and data["UseLinguistInfo"]["useLemma"]:
            res += 'A '
        if not data["UseLinguistInfo"]["usePOS"] and not data["UseLinguistInfo"]["useLemma"]:
            res += 'A\''
        if data["UseLinguistInfo"]["useSytax"]:
            res += 'B '
        if data["UseLinguistInfo"]["useBiGram"]:
            res += 'C '
        if data["UseLinguistInfo"]["useTriGram"]:
            res += 'D '
        if data["generateS0B2Bigram"]:
            res += 'E '
        if data["useTransitionHistory"]["transitionHistoryLength1"]:
            res += 'F '
        if data["useTransitionHistory"]["transitionHistoryLength2"]:
            res += 'G '
        if data["useTransitionHistory"]["transitionHistoryLength3"]:
            res += 'H '
        if data["S0B0Distance"]:
            res += 'I '
        if data["S0S1Distance"]:
            res += 'J '
        if data["useSecondBufferElement"]:
            res += 'K '
        if data["useDictionary"] or data["enhanceMerge"]:
            res += 'L '
        if data["useStackLength"]:
            res += 'M '
        if ["enableSingleMWE"]:
            res += 'N '
        print filename,' : ',res
    return res


def setConfig(configFile):
    data = json.load(open(configFile))
    usePOS = True if data["UseLinguistInfo"]["usePOS"] else True
    useLemma = True if data["UseLinguistInfo"]["useLemma"] else True
    useSyntax = True if data["UseLinguistInfo"]["useSytax"] else True
    useBiGram = True if data["UseLinguistInfo"]["useBiGram"] else True

    useTriGram = True if data["UseLinguistInfo"]["useTriGram"] else False
    generateS0B2Bigram = True if data["generateS0B2Bigram"] else False
    historyLength1 = True if data["useTransitionHistory"]["transitionHistoryLength1"] else False
    historyLength2 = True if data["useTransitionHistory"]["transitionHistoryLength2"] else False
    historyLength3 = True if data["useTransitionHistory"]["transitionHistoryLength3"] else False
    useS0B0Distance = True if data["S0B0Distance"] else False
    useS0S1Distance = True if data["S0S1Distance"] else False
    useB1 = True if data["useSecondBufferElement"] else False
    useLexic = True if data["useDictionary"] or data["enhanceMerge"] else False
    enhanceMerge = True if data["useDictionary"] or data["enhanceMerge"] else False
    useStackLength = True if data["useStackLength"] else False
    smartMWTDetection = True if ["enableSingleMWE"] else False


#getConfig('/Users/halsaied/PycharmProjects/NNIdentSys/Reports/Langs')
