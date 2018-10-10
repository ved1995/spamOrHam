import nltk
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
import random
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


saved_documents=open('/home/ved/Desktop/spam_or_ham/documents.pickle','rb')
documents=pickle.load(saved_documents)
saved_documents.close()

saved_word_features=open('/home/ved/Desktop/spam_or_ham/word_features.pickle','rb')
word_features=pickle.load(saved_word_features)
saved_word_features.close()

saved_featuresets=open('/home/ved/Desktop/spam_or_ham/featuresets.pickle','rb')
featuresets=pickle.load(saved_featuresets)
saved_featuresets.close()

random.shuffle(featuresets)
training_set=featuresets[:int(.8*len(featuresets))]
testing_set=featuresets[int(.8*len(featuresets)):]

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_MNB=open('/home/ved/Desktop/spam_or_ham/MNB_classifier.pickle','wb')
pickle.dump(MNB_classifier,save_MNB)
save_MNB.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_Bernoulli=open('/home/ved/Desktop/spam_or_ham/BernoulliNB_classifier.pickle','wb')
pickle.dump(BernoulliNB_classifier,save_Bernoulli)
save_Bernoulli.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_Logistic=open('/home/ved/Desktop/spam_or_ham/LogisticRegression_classifier.pickle','wb')
pickle.dump(LogisticRegression_classifier,save_Logistic)
save_Logistic.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_SGD=open('/home/ved/Desktop/spam_or_ham/SGDClassifier_classifier.pickle','wb')
pickle.dump(SGDClassifier_classifier,save_SGD)
save_SGD.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_LinearSVC=open('/home/ved/Desktop/spam_or_ham/LinearSVC_classifier.pickle','wb')
pickle.dump(LinearSVC_classifier,save_LinearSVC)
save_LinearSVC.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_NuSVC=open('/home/ved/Desktop/spam_or_ham/NuSVC_classifier.pickle','wb')
pickle.dump(NuSVC_classifier,save_NuSVC)
save_NuSVC.close()

voted_classifier=VoteClassifier (
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


print("accuracy of the voted_classifier is :", (nltk.classify.accuracy(voted_classifier,testing_set))*100)