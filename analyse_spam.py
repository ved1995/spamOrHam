from voting import spam_analysis
import io
with io.open('/home/ved/Desktop/email_data/enron1/ham/0003.1999-12-14.farmer.ham.txt', encoding ='latin-1') as f:
    file_data=f.read()
print(spam_analysis(file_data))