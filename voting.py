import nltk
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self.classifiers=classifiers
        
    def classify(self,feature):
        votes=[]
        for c in self.classifiers:
            vote=c.classify(feature)
            votes.append(vote)
        return mode(votes)

    def confidence(self,feature):
        votes=[]
        for c in self.classifiers:
            vote=c.classify(feature)
            votes.append(vote)
        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(votes)
        return conf

def create_features(document):
    featureset={}
    words=word_tokenize(document.lower())
    for w in word_features:
        featureset[w]=(w in words)
    return featureset

saved_word_features=open('/home/ved/Desktop/spam_or_ham/word_features.pickle','rb')
word_features=pickle.load(saved_word_features)
saved_word_features.close()

saved_featuresets=open('/home/ved/Desktop/spam_or_ham/featuresets.pickle','rb')
featuresets=pickle.load(saved_featuresets)
saved_featuresets.close()

testing_set=featuresets[int(.8*len(featuresets)):]


saved_classifier=open('/home/ved/Desktop/spam_or_ham/classifier.pickle','rb')
classifier=pickle.load(saved_classifier)
saved_classifier.close()


saved_MNB=open('/home/ved/Desktop/spam_or_ham/MNB_classifier.pickle','rb')
MNB_classifier=pickle.load(saved_MNB)
saved_MNB.close()


saved_Bernoulli=open('/home/ved/Desktop/spam_or_ham/BernoulliNB_classifier.pickle','rb')
BernoulliNB_classifier=pickle.load(saved_Bernoulli)
saved_Bernoulli.close()


saved_Logistic=open('/home/ved/Desktop/spam_or_ham/LogisticRegression_classifier.pickle','rb')
LogisticRegression_classifier=pickle.load(saved_Logistic)
saved_Logistic.close()


saved_SGD=open('/home/ved/Desktop/spam_or_ham/SGDClassifier_classifier.pickle','rb')
SGDClassifier_classifier=pickle.load(saved_SGD)
saved_SGD.close()



saved_LinearSVC=open('/home/ved/Desktop/spam_or_ham/LinearSVC_classifier.pickle','rb')
LinearSVC_classifier=pickle.load(saved_LinearSVC)
saved_LinearSVC.close()


saved_NuSVC=open('/home/ved/Desktop/spam_or_ham/NuSVC_classifier.pickle','rb')
NuSVC_classifier=pickle.load(saved_NuSVC)
saved_NuSVC.close()


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


def spam_analysis(feature):
    feature=feature.replace("\n","")
    feature=feature.replace("\t","")
    featureset=create_features(feature)
    evaluation=voted_classifier.classify(featureset)
    confidence=voted_classifier.confidence(featureset)
    return evaluation, confidence

