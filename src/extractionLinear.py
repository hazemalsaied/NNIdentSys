from config import configuration
from corpus import Token, getTokens, getTokenLemmas

mwtDictionary = {}
mweDictionary = {}


def extract(corpus):
    labels, featureDicss = [], []
    global mweDictionary, mwtDictionary
    mweDictionary, mwtDictionary = corpus.mweDictionary, corpus.mwtDictionary
    for sent in corpus.trainingSents:
        if configuration["sampling"]["importantSentences"] and not sent.vMWEs:
            continue
        l, f = extractSent(sent)
        labels.extend(l)
        featureDicss.extend(f)
    return labels, featureDicss


def extractSent(sent):
    transition = sent.initialTransition
    labels, features = [], []
    while transition.next:
        if transition.next and transition.next.type:
            labels.append(transition.next.type.value)
            features.append(getFeatures(transition, sent))
            transition = transition.next
    sent.featuresInfo = [labels, features]
    return labels, features


def getFeatures(transition, sent):
    featureDictionary = {}
    conf = transition.configuration
    dicFeat = configuration["features"]["dictionary"]
    if dicFeat["mwt"]:
        if conf.stack and isinstance(conf.stack[-1], Token) and conf.stack[-1].getLemma() in mwtDictionary:
            featureDictionary['S0_isMWT'] = True
            featureDictionary[conf.stack[-1].getLemma() + '_isMWT'] = True
            return featureDictionary
    # TODO return transDic directly in this case
    if configuration["features"]["stackLength"] and len(conf.stack) > 1:
        featureDictionary['StackLength'] = len(conf.stack)

    if len(conf.stack) >= 2:
        stackElements = [conf.stack[-2], conf.stack[-1]]
    else:
        stackElements = conf.stack

    # General linguistic Informations
    if stackElements:
        elemIdx = len(stackElements) - 1
        for elem in stackElements:
            generateLinguisticFeatures(elem, 'S' + str(elemIdx), featureDictionary)
            elemIdx -= 1

    if len(conf.buffer) > 0:
        generateLinguisticFeatures(conf.buffer[0], 'B0', featureDictionary)

        if configuration["features"]["unigram"]["b1"] and len(conf.buffer) > 1:
            generateLinguisticFeatures(conf.buffer[1], 'B1', featureDictionary)

    # Bi-Gram Generation
    if configuration["features"]["bigram"]["active"]:
        if len(stackElements) > 1:
            # Generate a Bi-gram S1S0 S0B0 S1B0 S0B1
            generateBiGram(stackElements[-2], stackElements[-1], 'S1S0', featureDictionary)
        if stackElements and conf.buffer:
            generateBiGram(stackElements[-1], conf.buffer[0], 'S0B0', featureDictionary)
            if len(stackElements) > 1:
                generateBiGram(stackElements[-2], conf.buffer[0], 'S1B0', featureDictionary)
            if len(conf.buffer) > 1:
                generateBiGram(stackElements[-1], conf.buffer[1], 'S0B1', featureDictionary)
                if configuration["features"]["bigram"]["s0b2"] and len(conf.buffer) > 2:
                    generateBiGram(stackElements[-1], conf.buffer[2], 'S0B2', featureDictionary)

    # Tri-Gram Generation
    if configuration["features"]["trigram"] and len(stackElements) > 1 and len(conf.buffer) > 0:
        generateTriGram(stackElements[-2], stackElements[-1], conf.buffer[0], 'S1S0B0', featureDictionary)

    # Syntaxic Informations
    if stackElements and configuration["features"]["syntax"]["active"]:
        generateSyntaxicFeatures(conf.stack, conf.buffer, featureDictionary)

    if stackElements and configuration["features"]["syntax"]["abstract"]:
        generateAbstractSyntaxicFeatures(conf.stack, conf.buffer, featureDictionary)

    # Distance information
    if configuration["features"]["distance"]["s0b0"] and conf.stack and conf.buffer:
        stackTokens = getTokens(conf.stack[-1])
        # sent.tokens.index(conf.buffer[0].position) - sent.tokens.index(stackTokens[-1])
        featureDictionary['S0B0Distance'] = str(conf.buffer[0].position - stackTokens[-1].position)
    if configuration["features"]["distance"]["s0s1"] and len(conf.stack) > 1 and isinstance(conf.stack[-1], Token) \
            and isinstance(conf.stack[-2], Token):
        featureDictionary['S0S1Distance'] = str(
            sent.tokens.index(conf.stack[-1]) - sent.tokens.index(conf.stack[-2]))
    addTransitionHistory(transition, featureDictionary)

    if configuration["features"]["dictionary"]["active"] and conf.buffer and conf.stack:
        generateDisconinousFeatures(conf, sent, featureDictionary)

        enhanceMerge(transition, featureDictionary)

    return featureDictionary


def enhanceMerge(transition, transDic):
    if not configuration["features"]["dictionary"]["active"]:
        return
    config = transition.configuration
    if transition.type and transition.type.value != 0 and len(config.buffer) > 0 and len(
            config.stack) > 0 and isinstance(config.stack[-1], Token):
        if isinstance(config.stack[-1], Token) and areInLexic([config.stack[-1], config.buffer[0]]):
            transDic['S0B0InLexicon'] = True
        if len(config.buffer) > 1 and areInLexic([config.stack[-1], config.buffer[0], config.buffer[1]]):
            transDic['S0B0B1InLexicon'] = True
        if len(config.buffer) > 2 and areInLexic(
                [config.stack[-1], config.buffer[0], config.buffer[1], config.buffer[2]]):
            transDic['S0B0B1B2InLexicon'] = True
        if len(config.buffer) > 1 and len(config.stack) > 1 and areInLexic(
                [config.stack[-2], config.stack[-1], config.buffer[1]]):
            transDic['S1S0B1InLexicon'] = True

    if len(config.buffer) > 0 and len(config.stack) > 1 and areInLexic(
            [config.stack[-2], config.buffer[0]]) and not areInLexic(
        [config.stack[-1], config.buffer[0]]):
        transDic['S1B0InLexicon'] = True
        transDic['S0B0tInLexicon'] = False
        if len(config.buffer) > 1 and areInLexic(
                [config.stack[-2], config.buffer[1]]) and not areInLexic(
            [config.stack[-1], config.buffer[1]]):
            transDic['S1B1InLexicon'] = True
            transDic['S0B1InLexicon'] = False


def generateDisconinousFeatures(configuration, sent, transDic):
    # print configuration.stack
    # if configuration.stack:
    #     print printStack(configuration.stack)
    # printStack([configuration.stack[-1]])
    tokens = getTokens([configuration.stack[-1]])
    tokenTxt = getTokenLemmas(tokens)
    for key in mweDictionary.keys():
        if tokenTxt in key and tokenTxt != key:
            bufidx = 0
            for bufElem in configuration.buffer[:5]:
                if bufElem.lemma != '' and (
                        (tokenTxt + ' ' + bufElem.lemma) in key or (bufElem.lemma + ' ' + tokenTxt) in key):
                    transDic['S0B' + str(bufidx) + 'AreMWETokens'] = True
                    transDic['S0B' + str(bufidx) + 'ArePartsOfMWEDistance'] = sent.tokens.index(
                        bufElem) - sent.tokens.index(tokens[-1])
                bufidx += 1
            break


def generateLinguisticFeatures(stackElem, label, featureDictionary):
    if isinstance(stackElem, list):
        stackElemToken = concatenateTokens(stackElem)[0]
    else:
        stackElemToken = concatenateTokens([stackElem])[0]
    if configuration["features"]["unigram"]["token"]:
        featureDictionary[label + 'Token'] = stackElemToken.text
    if configuration["features"]["unigram"]["pos"] and stackElemToken.posTag and stackElemToken.posTag.strip() != '':
        featureDictionary[label + 'POS'] = stackElemToken.posTag
    if configuration["features"]["unigram"]["lemma"] and stackElemToken.lemma and stackElemToken.lemma.strip() != '':
        featureDictionary[label + 'Lemma'] = stackElemToken.lemma
    if configuration["features"]["unigram"]["suffix"]:
        featureDictionary[label + '_LastThreeLetters'] = stackElemToken.text[-3:]
        featureDictionary[label + '_LastTwoLetters'] = stackElemToken.text[-2:]
        # @TODO reintegrer
        # if v2featureSettings.useDictionary and ((
        # token.lemma != '' and token.lemma in Corpus.mweTokenDic.keys()) or token.text in Corpus.mweTokenDic.keys()):
        #     transDic[label + 'IsInLexic'] = 'true'


def generateSyntaxicFeatures(stack, buffer, dic):
    if stack:
        stack0 = stack[-1]
        if not isinstance(stack0, Token):
            return
        if int(stack0.dependencyParent) == -1 or int(
                stack0.dependencyParent) == 0 or stack0.dependencyLabel.strip() == '' or not buffer and not buffer:
            return
        for bElem in buffer:
            if bElem.dependencyParent == stack0.position:
                dic['hasRighDep_' + bElem.dependencyLabel] = 'true'
                dic[stack0.getLemma() + '_hasRighDep_' + bElem.dependencyLabel] = 'true'
                dic[stack0.getLemma() + '_' + bElem.getLemma() + '_hasRighDep_' + bElem.dependencyLabel] = 'true'

        if stack0.dependencyParent > stack0.position:
            for bElem in buffer:
                if bElem.position == stack0.dependencyParent:
                    dic[stack0.lemma + '_isGouvernedBy_' + bElem.getLemma()] = 'true'
                    dic[stack0.lemma + '_isGouvernedBy_' + bElem.getLemma() + '_' + stack0.dependencyLabel] = 'true'
                    break
        if len(stack) > 1:
            stack1 = stack[-2]
            if not isinstance(stack1, Token):
                return
            if stack0.dependencyParent == stack1.position:
                dic['SyntaxicRelation'] = '+' + stack0.dependencyLabel
            elif stack0.position == stack1.dependencyParent:
                dic['SyntaxicRelation'] = '-' + stack1.dependencyLabel


def generateAbstractSyntaxicFeatures(stack, buffer, dic):
    if stack:
        stack0 = stack[-1]
        if not isinstance(stack0, Token):
            return
        if int(stack0.dependencyParent) == -1 or int(
                stack0.dependencyParent) == 0 or stack0.dependencyLabel.strip() == '' or not buffer and not buffer:
            return
        for bElem in buffer:
            if bElem.dependencyParent == stack0.position:
                bIdx = buffer.index(bElem)
                dic['RighDep(S0)'] = bElem.dependencyLabel
                dic['RighDep(S0)'] = 'B{0}'.format(bIdx)
                dic['RighDep(S0,B{0})'.format(bIdx)] = bElem.dependencyLabel

        if stack0.dependencyParent > stack0.position:
            for bElem in buffer:
                bIdx = buffer.index(bElem)
                if bElem.position == stack0.dependencyParent:
                    dic['Gouverner(S0)'] = bIdx
                    dic['Gouvernence(B{0}, S0)'.format(bIdx)] = stack0.dependencyLabel
                    break
        if len(stack) > 1:
            stack1 = stack[-2]
            if not isinstance(stack1, Token):
                return
            if stack0.dependencyParent == stack1.position:
                dic['SyntaxicRelation(S0,S1)'] = '+' + stack0.dependencyLabel
            elif stack0.position == stack1.dependencyParent:
                dic['SyntaxicRelation(S0,S1)'] = '-' + stack1.dependencyLabel


def generateTriGram(token0, token1, token2, label, transDic):
    tokens = concatenateTokens([token0, token1, token2])
    getFeatureInfo(transDic, label + 'Token', tokens, 'ttt')
    getFeatureInfo(transDic, label + 'Lemma', tokens, 'lll')
    getFeatureInfo(transDic, label + 'POS', tokens, 'ppp')
    getFeatureInfo(transDic, label + 'LemmaPOSPOS', tokens, 'lpp')
    getFeatureInfo(transDic, label + 'POSLemmaPOS', tokens, 'plp')
    getFeatureInfo(transDic, label + 'POSPOSLemma', tokens, 'ppl')
    getFeatureInfo(transDic, label + 'LemmaLemmaPOS', tokens, 'llp')
    getFeatureInfo(transDic, label + 'LemmaPOSLemma', tokens, 'lpl')
    getFeatureInfo(transDic, label + 'POSLemmaLemma', tokens, 'pll')


def generateBiGram(token0, token1, label, transDic):
    tokens = concatenateTokens([token0, token1])
    getFeatureInfo(transDic, label + 'Token', tokens, 'tt')
    getFeatureInfo(transDic, label + 'Lemma', tokens, 'll')
    getFeatureInfo(transDic, label + 'POS', tokens, 'pp')
    getFeatureInfo(transDic, label + 'LemmaPOS', tokens, 'lp')
    getFeatureInfo(transDic, label + 'POSLemma', tokens, 'pl')


def concatenateTokens(elements):
    tokenDic, tokens = {}, []
    for idx in range(len(elements)):
        tokenDic[idx] = Token(-1, '', lemma='', posTag='')
        for subToken in getTokens(elements[idx]):
            tokenDic[idx].text += subToken.text + '_'
            tokenDic[idx].lemma += subToken.lemma + '_'
            tokenDic[idx].posTag += subToken.posTag + '_'
        tokenDic[idx].text = tokenDic[idx].text[:-1]
        tokenDic[idx].lemma = tokenDic[idx].lemma[:-1]
        tokenDic[idx].posTag = tokenDic[idx].posTag[:-1]
        tokens.append(tokenDic[idx])
    return tokens


def getFeatureInfo(dic, label, tokens, features):
    idx, feature = 0, ''
    if not configuration["features"]["unigram"]["token"] and 't' in features:
        return
    if not configuration["features"]["unigram"]["lemma"] and 'l' in features:
        return
    if not configuration["features"]["unigram"]["pos"] and 'p' in features:
        return

    for token in tokens:
        if features[idx].lower() == 'l':
            if token.lemma.strip() != '':
                feature += token.lemma.strip() + '_'
            else:
                feature += '*' + '_'
        elif features[idx].lower() == 'p':
            if token.posTag.strip() != '':
                feature += token.posTag.strip() + '_'
            else:
                feature += '*' + '_'
        elif features[idx].lower() == 't':
            if token.text.strip() != '':
                feature += token.text.strip() + '_'
        idx += 1
    if len(feature) > 0:
        feature = feature[:-1]
        dic[label] = feature
    return ''


def areInLexic(tokensList):
    if getTokenLemmas(tokensList) in mweDictionary.keys():
        return True
    return False


def addTransitionHistory(transition, transDic):
    if configuration["features"]["history"]["1"]:
        getTransitionHistory(transition, 1, 'TransHistory1', transDic)
    if configuration["features"]["history"]["2"]:
        getTransitionHistory(transition, 2, 'TransHistory2', transDic)
    if configuration["features"]["history"]["3"]:
        getTransitionHistory(transition, 3, 'TransHistory3', transDic)


def getTransitionHistory(transition, length, label, transDic):
    idx, history = 0, ''
    transition = transition.previous
    while transition and idx < length:
        if transition.type:
            history += str(transition.type.value)
        transition = transition.previous
        idx += 1
    if len(history) == length:
        transDic[label] = history
