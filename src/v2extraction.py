from corpus import Token, getTokens, getTokenLemmas

# @TODO
mwtDictionary = {}
mweDictionary = {}


def extract(corpus):
    labels, featureDicss = [], []
    global mweDictionary, mwtDictionary
    mweDictionary, mwtDictionary = corpus.mweDictionary, corpus.mwtDictionary
    for sent in corpus.trainingSents:
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

    if featureSettings.smartMWTDetection:
        if conf.stack and isinstance(conf.stack[-1], Token) and conf.stack[-1].getLemma() in mwtDictionary:
            featureDictionary['isMWT_' + mwtDictionary[conf.stack[-1].getLemma()].lower()] = True
    # TODO return transDic directly in this case
    if featureSettings.useStackLength and len(conf.stack) > 1:
        featureDictionary['StackLengthIs'] = len(conf.stack)

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
        if featureSettings.useFirstBufferElement:
            generateLinguisticFeatures(conf.buffer[0], 'B0', featureDictionary)

        if featureSettings.useSecondBufferElement and len(conf.buffer) > 1:
            generateLinguisticFeatures(conf.buffer[1], 'B1', featureDictionary)

    # Bi-Gram Generation
    if featureSettings.useBiGram:
        if len(stackElements) > 1:
            # Generate a Bi-gram S1S0 S0B0 S1B0 S0B1
            generateBiGram(stackElements[-2], stackElements[-1], 'S1S0', featureDictionary)
            if featureSettings.generateS1B1 and len(conf.buffer) > 1:
                generateBiGram(stackElements[-2], conf.buffer[1], 'S1B1', featureDictionary)
        if stackElements and conf.buffer:
            generateBiGram(stackElements[-1], conf.buffer[0], 'S0B0', featureDictionary)
            if len(stackElements) > 1:
                generateBiGram(stackElements[-2], conf.buffer[0], 'S1B0', featureDictionary)
            if len(conf.buffer) > 1:
                generateBiGram(stackElements[-1], conf.buffer[1], 'S0B1', featureDictionary)
                if featureSettings.generateS0B2Bigram and len(conf.buffer) > 2:
                    generateBiGram(stackElements[-1], conf.buffer[2], 'S0B2', featureDictionary)

    # Tri-Gram Generation
    if featureSettings.useTriGram and len(stackElements) > 1 and len(conf.buffer) > 0:
        generateTriGram(stackElements[-2], stackElements[-1], conf.buffer[0], 'S1S0B0', featureDictionary)

    # Syntaxic Informations
    if stackElements and featureSettings.useSyntax:
        generateSyntaxicFeatures(conf.stack, conf.buffer, featureDictionary)

    # Distance information
    if featureSettings.useS0B0Distance and conf.stack and conf.buffer:
        stackTokens = getTokens(conf.stack[-1])
        # sent.tokens.index(conf.buffer[0].position) - sent.tokens.index(stackTokens[-1])
        featureDictionary['S0B0Distance'] = str(conf.buffer[0].position - stackTokens[-1].position)
    if featureSettings.useS0S1Distance and len(conf.stack) > 1 and isinstance(conf.stack[-1], Token) \
            and isinstance(conf.stack[-2], Token):
        featureDictionary['S0S1Distance'] = str(
            sent.tokens.index(conf.stack[-1]) - sent.tokens.index(conf.stack[-2]))
    addTransitionHistory(transition, featureDictionary)

    if featureSettings.useLexic and conf.buffer and conf.stack:
        generateDisconinousFeatures(conf, sent, featureDictionary)

    enhanceMerge(transition, featureDictionary)

    return featureDictionary


def enhanceMerge(transition, transDic):
    if not featureSettings.enhanceMerge:
        return
    config = transition.configuration
    if transition.type and transition.type.value != 0 and len(config.buffer) > 0 and len(
            config.stack) > 0 and isinstance(config.stack[-1], Token):
        if isinstance(config.stack[-1], Token) and areInLexic([config.stack[-1], config.buffer[0]]):
            transDic['S0B0InLexic'] = True
        if len(config.buffer) > 1 and areInLexic([config.stack[-1], config.buffer[0], config.buffer[1]]):
            transDic['S0B0B1InLexic'] = True
        if len(config.buffer) > 2 and areInLexic(
                [config.stack[-1], config.buffer[0], config.buffer[1], config.buffer[2]]):
            transDic['S0B0B1B2InLexic'] = True
        if len(config.buffer) > 1 and len(config.stack) > 1 and areInLexic(
                [config.stack[-2], config.stack[-1], config.buffer[1]]):
            transDic['S1S0B1InLexic'] = True

    if len(config.buffer) > 0 and len(config.stack) > 1 and areInLexic(
            [config.stack[-2], config.buffer[0]]) and not areInLexic(
        [config.stack[-1], config.buffer[0]]):
        transDic['S1B0InLexic'] = True
        transDic['S0B0tInLexic'] = False
        if len(config.buffer) > 1 and areInLexic(
                [config.stack[-2], config.buffer[1]]) and not areInLexic(
            [config.stack[-1], config.buffer[1]]):
            transDic['S1B1InLexic'] = True
            transDic['S0B1InLexic'] = False


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
                    transDic['S0B' + str(bufidx) + 'ArePartsOfMWE'] = True
                    transDic['S0B' + str(bufidx) + 'ArePartsOfMWEDistance'] = sent.tokens.index(
                        bufElem) - sent.tokens.index(tokens[-1])
                bufidx += 1
            break


def generateLinguisticFeatures(token, label, featureDictionary):
    if isinstance(token, list):
        token = concatenateTokens(token)[0]
    else:
        token = concatenateTokens([token])[0]
    featureDictionary[label + 'Token'] = token.text
    if featureSettings.usePOS and token.posTag and token.posTag.strip() != '':
        featureDictionary[label + 'POS'] = token.posTag
    if featureSettings.useLemma and token.lemma and token.lemma.strip() != '':
        featureDictionary[label + 'Lemma'] = token.lemma
    if not featureSettings.useLemma and not featureSettings.usePOS:
        featureDictionary[label + '_LastThreeLetters'] = token.text[-3:]
        featureDictionary[label + '_LastTwoLetters'] = token.text[-2:]
        # @TODO reintegrer
        # if featureSettings.useDictionary and ((
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
    for token in tokens:
        if features[idx].lower() == 'l':
            if featureSettings.useLemma:
                if token.lemma.strip() != '':
                    feature += token.lemma.strip() + '_'
                else:
                    feature += '*' + '_'
        elif features[idx].lower() == 'p':
            if featureSettings.usePOS:
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
    if featureSettings.historyLength1:
        getTransitionHistory(transition, 1, 'TransHistory1', transDic)
    if featureSettings.historyLength2:
        getTransitionHistory(transition, 2, 'TransHistory2', transDic)
    if featureSettings.historyLength3:
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
