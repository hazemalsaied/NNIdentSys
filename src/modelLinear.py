import sys
from datetime import datetime

from imblearn.over_sampling import RandomOverSampler
from extractionLinear import extract, getFeatures
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import  LogisticRegression
from transitions import *
import reports

def train(corpus, svm=False):
    sys.stdout.write('Linear model \n')
    labels, featureDicss = extract(corpus)
    startTime = datetime.now()
    vec = DictVectorizer()
    features = vec.fit_transform(featureDicss)
    sys.stdout.write('# Feature number = {0}\n'.format(len(vec.vocabulary_)))
    if svm:
        clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
    else:
        clf = LogisticRegression(solver='sag')
    sys.stdout.write('# model = LinearSVC\n')
    if configuration["sampling"]["overSampling"]:
        sys.stdout.write('Train data = {0}, '.format(features.shape[0]))
        ros = RandomOverSampler(random_state=0)
        features, labels = ros.fit_sample(features, labels)
        sys.stdout.write('Train data = {0}, '.format(features.shape[0]))

    # clf.fit(features.toarray(), labels) .tocsr()
    clf.fit(features, labels)
    sys.stdout.write(reports.doubleSep + reports.tabs + 'Training time : {0}'.format(datetime.datetime.now() - time)
                     + reports.doubleSep)
    return clf, vec


def parse(corpus, clf, vectoeizer, printt=False):
    initializeSent(corpus)
    debug, printedNum = False, 0
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
        if sent.vMWEs and printt and printedNum < 10:
            print sent
            printedNum += 1


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

