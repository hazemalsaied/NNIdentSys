import sys

from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OutputCodeClassifier
from sklearn.svm import LinearSVC

from extractionLinear import extract
from reports import tabs, seperator
from transitions import *
from sklearn import svm

def train(corpus):
    linearConf = configuration['linear']
    labels, featureDicss = extract(corpus)
    vec = DictVectorizer()
    features = vec.fit_transform(featureDicss)
    sys.stdout.write(tabs + 'Feature number = {0}\n'.format(len(vec.vocabulary_))
                     + seperator)
    # svm_clf = SVC(probability=True)
    # clf = OneVsRestClassifier(LinearSVC(random_state=0)) if linearConf['svm'] else LogisticRegression(solver='sag')
    # OneVsRestClassifier(LinearSVC(random_state=0))
    clf = svm.SVC(kernel='linear', C = 1) if linearConf['svm'] else \
        LogisticRegression(solver='sag')
    # clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0) if linearConf['svm'] else \
    #    LogisticRegression(solver='sag')
    sys.stdout.write(str(clf) + '\n')
    if configuration["sampling"]["overSampling"]:
        sys.stdout.write('Train data = {0}, '.format(features.shape[0]))
        ros = RandomOverSampler(random_state=0)
        features, labels = ros.fit_sample(features, labels)
        sys.stdout.write('Train data = {0}, '.format(features.shape[0]))
    clf.fit(features, labels)
    # clf.fit(features.toarray() if linearConf['svm'] else features, labels)
    return clf, vec
