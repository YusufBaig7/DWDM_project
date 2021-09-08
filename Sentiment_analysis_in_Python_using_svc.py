import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def clean_rev(rev, remove_stopwords =True):
    rev = rev.lower().split()
    
    if remove_stopwords:
        stop = set(stopwords.words("english"))
        rev = [w for w in rev if not w in stop]
    rev = " ".join(rev)
    
    rev = re.sub(r"<br />", " ", rev)
    rev = re.sub(r"[^a-z]", " ", rev)
    rev = re.sub(r"   ", " ", rev) 
    rev = re.sub(r"  ", " ", rev)
    
    return (rev)

df = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.head()

df['review'] = df['review'].apply(clean_rev)
df.head()

encoder = LabelEncoder()
df['sentiment'] = encoder.fit_transform(df['sentiment'])
df.head()

rev = df['review']
sent = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(rev, sent, test_size = 0.2, random_state = 0)

train = []
test  = []

for i in x_train.index:
    temp=x_train[i]
    train.append(temp)

for j in x_test.index:
    temp1=x_test[j]
    test.append(temp1)

cv = CountVectorizer()
cv_train = cv.fit_transform(train)
cv_test = cv.transform(test)

svc=LinearSVC(random_state= 0 ,max_iter=15000)
svc.fit(cv_train,y_train)

y_pred=svc.predict(cv_test)

print(classification_report(y_test, y_pred))
print("Accuracy is",accuracy_score(y_test, y_pred))