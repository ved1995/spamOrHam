import os
import io
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pickle
root_directory="/home/ved/Desktop/email_data/"
all_words=[]
all_word_type=["J","R","V"]

ham_list=[]
spam_list=[]
for directories, subdirectories, filenames in os.walk(root_directory):
     if os.path.split(directories)[1]=='ham':
        for fileName in filenames:
            with io.open(os.path.join(directories,fileName),encoding="latin-1") as f:
                files_ham=f.read()
                
                files_ham=files_ham.replace("\t","")
                files_ham=files_ham.replace("\n","")
                words=word_tokenize(files_ham)
                pos=nltk.pos_tag(words)
                for p in pos:
                    if p[1][0] in all_word_type:
                        all_words.append(p[0].lower())
                # for w in words:
                #     all_words.append(w)
                ham_list.append((files_ham,"ham"))
                
     if os.path.split(directories)[1]=='spam':
        for fileName in filenames:
            with io.open(os.path.join(directories,fileName), encoding="latin-1") as f:
                files_spam=f.read()
                 
                files_spam=files_spam.replace("\t","")
                files_spam=files_spam.replace("\n","")
                words=word_tokenize(files_spam)
                pos=nltk.pos_tag(words)
                for p in pos:
                    if p[1][0] in all_word_type:
                        all_words.append(p[0].lower())
                # for w in words:
                #     all_words.append(w)
                spam_list.append((files_spam,"spam"))

word_frequency=nltk.FreqDist(all_words)
word_features=list(word_frequency.keys())[:5000]
save_word_feature=open('/home/ved/Desktop/spam_or_ham/word_features.pickle','wb')
pickle.dump(word_features,save_word_feature)
save_word_feature.close()

documents=[]
for l in ham_list:
    documents.append(l)
for l in spam_list:
    documents.append(l)

save_documents=open('/home/ved/Desktop/spam_or_ham/documents.pickle','wb')
pickle.dump(documents, save_documents)
save_documents.close()

print('successfully saved documents')

        


