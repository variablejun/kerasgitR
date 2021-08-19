import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm

from konlpy.tag import Okt

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

train = pd.read_csv('../_data/dacon/train.csv')
test = pd.read_csv('../_data/dacon/test.csv')
sample_submission=pd.read_csv('../_data/dacon/sample_submission.csv')

print(train.shape)
print(test.shape)

'''
(174304, 13)
(43576, 12)
'''
train=train[['과제명','label']]
test=test[['과제명']]

def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
     text=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]","", text)
     word_text=okt.morphs(text, stem=True)
     if remove_stopwords:
        word_review=[token for token in word_text if not token in stop_words]
     return word_review

stop_words=['은','는','이','가', '하','아','것','들','의','있','되','수','보','주','등','한']
okt=Okt()
clean_train_text=[]
clean_test_text=[]

for text in tqdm.tqdm(train['과제명']):
     try:
        clean_train_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
     except:
        clean_train_text.append([])

for text in tqdm.tqdm(test['과제명']):
    if type(text) == str:
        clean_test_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text.append([])
           
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(tokenizer = lambda x: x, lowercase=False)
train_features=vectorizer.fit_transform(clean_train_text)
test_features=vectorizer.transform(clean_test_text)    

TEST_SIZE=0.2
RANDOM_SEED=42

train_x, eval_x, train_y, eval_y = train_test_split(train_features, train['label'], test_size=TEST_SIZE, random_state=RANDOM_SEED)

np.save('./_save/x_train.npy',arr=train_x)

np.save('./_save/eval_x.npy',arr=eval_x)

np.save('./_save/train_y.npy',arr=train_y)

np.save('./_save/eval_y.npy',arr=eval_y)
