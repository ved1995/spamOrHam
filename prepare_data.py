import nltk
from nltk.tokenize import word_tokenize
import pickle
from sklearn.naive_bayes import MultinomialNB
import random

saved_word_features=open('/home/ved/Desktop/spam_or_ham/word_features.pickle','rb')
word_features=pickle.load(saved_word_features)
saved_word_features.close()
print('got saved_word_features \n')

saved_documents=open('/home/ved/Desktop/spam_or_ham/documents.pickle','rb')
documents=pickle.load(saved_documents)
saved_documents.close()
print('got saved_documents ...\n')

def create_features(document):
    featureset={}
    words=word_tokenize(document.lower())
    for w in word_features:
        featureset[w]=(w in words)
    return featureset

featuresets=[ (create_features(doc),category) for (doc,category) in documents]
save_featuresets=open('/home/ved/Desktop/spam_or_ham/featuresets.pickle','wb')
pickle.dump(featuresets,save_featuresets)
save_featuresets.close()

print('featuresets created successfully \n')

random.shuffle(featuresets)
training_set=featuresets[:int(.8*len(featuresets))]
testing_set=featuresets[int(.8*len(featuresets)):]

classifier=nltk.NaiveBayesClassifier.train(training_set)
print('original classifier accuracy is :' , (nltk.classify.accuracy(classifier,testing_set))*100)

saved_classifier=open('/home/ved/Desktop/spam_or_ham/classifier.pickle','wb')
pickle.dump(classifier,saved_classifier)
saved_classifier.close()

print('\n classifier saved successfully')





    