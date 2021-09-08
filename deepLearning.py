import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import pandas as pd
import tensorflow.keras.datasets.imdb as imdb
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 

df = pd.read_csv('E:\Code\IMDB sentiment analysis\IMDB Dataset.csv')
df.head()

pre = df['review']
tweet = []
for item in pre:
    tweet.append(clean_tweet(item))

sent = df['sentiment']
sentiment = []
for item in sent:
    sentiment.append(item)

sentiment[0:5]

tweet[0:5]

vocab = 40000
max_length = 500
embedding_dim = 128
trunc_type = 'post'
pad_type = 'post'
oov_tok = "<OOV>"
training_size = 40000

training_tweets = tweet[0:training_size]
training_labels = sentiment[0:training_size]
test_sentence = tweet[training_size:]
test_label = sentiment[training_size:]

tokenizer = Tokenizer(num_words = vocab, oov_token = oov_tok)
tokenizer.fit_on_texts(training_tweets)

word_index = tokenizer.word_index

train_seq = tokenizer.texts_to_sequences(training_tweets)
train_pad = pad_sequences(train_seq, maxlen = max_length, padding = pad_type, truncating = trunc_type)

test_seq = tokenizer.texts_to_sequences(test_sentence)
test_pad = pad_sequences(test_seq, maxlen = max_length, padding = pad_type, truncating = trunc_type)

train = np.array(train_pad)
train_label = np.array(training_labels)
test = np.array(test_pad)
test_label = np.array(test_label)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

model = Sequential()
model.add(Embedding(vocab, embedding_dim, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer = 'Adam', metrics =['accuracy'] )
model.summary()

num_epoch = 50
model.fit(train, train_label, epochs=num_epoch, verbose = 1)