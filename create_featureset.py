import os
import io
import nltk
from nltk.tokenize import word_tokenize
root_directory="/home/ved/Desktop/email_data/"
all_words=[]

def find_feature(document):
    words=word_tokenize(document.lower())
    featureset={}
    for word in word_features:
        featureset[word]=(word in words) 
    return featureset

ham_list=[]
spam_list=[]
for directories, subdirectories, filenames in os.walk(root_directory):
     if os.path.split(directories)[1]=='ham':
        for fileName in filenames:
            with io.open(os.path.join(directories,fileName),encoding="latin-1") as f:
                files_ham=f.read()
                files=files_ham.split('\n')
                for l in files:
                    ham_list.append(l)
     if os.path.split(directories)[1]=='spam':
        for fileName in filenames:
            with io.open(os.path.join(directories,fileName), encoding="latin-1") as f:
                files_spam=f.read()
                files=files_spam.split('\n')
                for l in files:
                    spam_list.append(l)

for line in ham_list:
    words=word_tokenize(line)
    for w in words:
        all_words.append(w.lower())

for line in spam_list:
    words=word_tokenize(line)
    for w in words:
        all_words.append(w.lower())


all_words=nltk.FreqDist(all_words)
word_features=list(all_words.keys())[:5000]

featuresets=[]
for line in ham_list:
    featuresets.append((find_feature(line),"ham"))

for line in spam_list:
    featuresets.append((find_feature(line),"spam"))

training_set=featuresets[:int(len(featuresets)*.8)]
testing_set=featuresets[int(len(featuresets)*.8):]


print(training_set[:20])
print(testing_set[:20])

