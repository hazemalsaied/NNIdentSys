import sys
from datetime import datetime

from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

from transitions import *
from linearFeatures import extract, getFeatures


def train(corpus):
    labels, featureDicss = extract(corpus)
    startTime = datetime.now()

    vec = DictVectorizer()
    features = vec.fit_transform(featureDicss)
    sys.stdout.write('# Feature number = {0}\n'.format(len(vec.vocabulary_)))
    clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
    sys.stdout.write('# model = LinearSVC\n')
    clf.fit(features, labels)
    sys.stdout.write('# Training time = {0} minutes!\n'.format(str(datetime.now().minute - startTime.minute)))
    return clf, vec


def parse(corpus, clf, vectoeizer):
    initializeSent(corpus)
    printt, debug = False, False
    for sent in corpus.testingSents:
        if len(sent.vMWEs) >= 1:
            debug = True
        sent.initialTransition = Transition(None, isInitial=True, sent=sent)
        transition = sent.initialTransition
        while not transition.isTerminal():
            if len(transition.configuration.stack) == 2 and debug:
                pass
            newTransition = nextTrans(transition, sent, clf, vectoeizer)
            newTransition.apply(transition, sent, parse=True)
            transition = newTransition
        debug = False
        if len(sent.vMWEs) >= 1 and printt:
            print sent
            printt = False


def nextTrans(transition, sent, clf, vectorizer):
    legalTansDic = transition.getLegalTransDic()
    if len(legalTansDic) == 1:
        return initialize(legalTansDic.keys()[0], sent)
    featDic = getFeatures(transition, sent)
    transTypeValue = clf.predict(vectorizer.transform(featDic))[0]
    transType = getType(transTypeValue)
    if transType in legalTansDic:
        return legalTansDic[transType]
    if len(legalTansDic):
        return initialize(legalTansDic.keys()[0], sent)


def initializeSent(corpus):
    for sent in corpus.testingSents:
        sent.identifiedVMWEs = []
        sent.initialTransition = None
