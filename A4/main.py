# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import sys
from tensorflow import keras
from keras.models import Sequential
from keras import Input 
from keras.layers import Dense 
import pandas as pd 
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 

# add sys arg command line arguements here
print(sys.arg)
dataPath = sys.argv[1]

with open(dataPath + '/dataPath.txt', 'w') as f:
  f.write(dataPath)

# Positive Training Reviews without stopword removal
posTrainingReviews = dataPath + '/trainPos.csv'
columns = ['text']
posTrainingDf = pd.read_csv(posTrainingReviews, delimiter = "\t", names=columns)
posTrainingDf['sentient'] = 1
posTrainingDf['text'] = posTrainingDf['text'].str.replace('\d+', '')
posTrainingDf['text'] = posTrainingDf['text'].apply(eval).apply(' '.join)

# Positive Training Reviews with stopword removal
posTrainingReviewsNS = dataPath + '/train_nsPos.csv'
columns = ['text']
posTrainingDfNs = pd.read_csv(posTrainingReviewsNS, delimiter = "\t", names=columns)
posTrainingDfNs['sentient'] = 1
posTrainingDfNs['text'] = posTrainingDfNs['text'].str.replace('\d+', '')
posTrainingDfNs['text'] = posTrainingDfNs['text'].apply(eval).apply(' '.join)

# Negative Training Reviews without stopword removal
negTrainingReviews = dataPath + '/trainNeg.csv'
columns = ['text']
negTrainingDf = pd.read_csv(negTrainingReviews, delimiter = "\t", names=columns)
negTrainingDf['sentient'] = 0
negTrainingDf['text'] = negTrainingDf['text'].str.replace('\d+', '')
negTrainingDf['text'] = negTrainingDf['text'].apply(eval).apply(' '.join)

# Negative Training Reviews with stopword removal
negTrainingReviewsNS = dataPath + '/test_nsNeg.csv'
columns = ['text']
negTrainingDfNs = pd.read_csv(negTrainingReviewsNS, delimiter = "\t", names=columns)
negTrainingDfNs['sentient'] = 0
negTrainingDfNs['text'] = negTrainingDfNs['text'].str.replace('\d+', '')
negTrainingDfNs['text'] = negTrainingDfNs['text'].apply(eval).apply(' '.join)

# Positive Testing Reviews without stopword removal
posTestingReviews = dataPath + '/testPos.csv'
columns = ['text']
posTestingDf = pd.read_csv(posTestingReviews, delimiter = "\t", names=columns)
posTestingDf['sentient'] = 1
posTestingDf['text'] = posTestingDf['text'].str.replace('\d+', '')
posTestingDf['text'] = posTestingDf['text'].apply(eval).apply(' '.join)

# Positive Testing Reviews with stopword removal
posTestingReviewsNS = dataPath + '/test_nsPos.csv'
columns = ['text']
posTestingDfNs = pd.read_csv(posTestingReviewsNS, delimiter = "\t", names=columns)
posTestingDfNs['sentient'] = 1
posTestingDfNs['text'] = posTestingDfNs['text'].str.replace('\d+', '')
posTestingDfNs['text'] = posTestingDfNs['text'].apply(eval).apply(' '.join)

# Negative Testing Reviews without stopword removal
negTestingReviews = dataPath + '/testNeg.csv'
columns = ['text']
negTestingDf = pd.read_csv(negTestingReviews, delimiter = "\t", names=columns)
negTestingDf['sentient'] = 0
negTestingDf['text'] = negTestingDf['text'].str.replace('\d+', '')
negTestingDf['text'] = negTestingDf['text'].apply(eval).apply(' '.join)

# Negative Testing Reviews with stopword removal
negTestingReviewsNS = dataPath + '/test_nsNeg.csv'
columns = ['text']
negTestingDfNs = pd.read_csv(negTestingReviewsNS, delimiter = "\t", names=columns)
negTestingDfNs['sentient'] = 0
negTestingDfNs['text'] = negTestingDfNs['text'].str.replace('\d+', '')
negTestingDfNs['text'] = negTestingDfNs['text'].apply(eval).apply(' '.join)

#Combine pos and neg reviews without stopword removal
reviewsFrames = [posTrainingDf, negTrainingDf]
trainReviews = pd.concat(reviewsFrames)

#Combine pos and neg reviews WITH stopword removal
reviewsFramesNS = [posTrainingDfNs, negTrainingDfNs]
trainReviewsNS = pd.concat(reviewsFramesNS)

#Combine pos and neg reviews without stopword removal
reviewsFramesTesting = [posTestingDf, negTestingDf]
testReviews = pd.concat(reviewsFramesTesting)

#Combine pos and neg reviewws WITH stopword removal
reviewsFramesTestingNS = [posTestingDfNs, negTestingDfNs]
testReviewsNS = pd.concat(reviewsFramesTestingNS)

from nltk.util import pad_sequence
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import pickle

reviewsArray = []
for i in trainReviewsNS['text']:
  li = list(i.split(" "))
  reviewsArray.append(li)

reviewsArrayTest = []
for i in testReviewsNS['text']:
  li = list(i.split(" "))
  reviewsArrayTest.append(li)

wordToVecModel = Word2Vec(sentences=reviewsArray, window=5, min_count=1, size=300, workers=4)

word_vectors = wordToVecModel.wv
word_vectors.save("word2vec.wordvectors")
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
embeddings = wordToVecModel.wv.get_keras_embedding(train_embeddings=False)
vocabSize = wordToVecModel.wv.vectors.shape
print(wordToVecModel)
pickle.dump(wordToVecModel, open(dataPath + '/word2VecModel.pkl', 'wb'))

encodedReviews = [wordToVecModel.wv[word] for word in reviewsArray]
encodedReviews = pad_sequences(encodedReviews, maxlen=86, padding='post', truncating='post')

encodedReviewsTest = [wordToVecModel.wv[word] for word in reviewsArrayTest]
encodedReviewsTest = pad_sequences(encodedReviewsTest, maxlen=86, padding='post', truncating='post')

from os import name
import tensorflow as tf
from keras import layers
from keras import regularizers
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Flatten

vectorizer = CountVectorizer()
vectorizer.fit(trainReviewsNS)

yTrain = trainReviews['sentient']

yTest = testReviewsNS['sentient']

#reluModel Creation
reluModel = Sequential(name='reluModel')
embeddingLayer = Embedding(4,32,input_length=86)
reluModel.add(embeddingLayer)
reluModel.add(Flatten())
reluModel.add(Dense(128,activation='relu', name='hiddenLayer'))
reluModel.add(Dense(2, activation='softmax', name='outputLayer'))
L2layer = tf.keras.layers.Dense(5, kernel_initializer='ones',
                              activity_regularizer=tf.keras.regularizers.l2(0.01),
                              name='L2Layer')
reluModel.add(L2layer)
reluModel.add(Dropout(0.2, input_shape=(60,)))


reluModel.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    loss_weights=None, 
    weighted_metrics=None, 
    run_eagerly=None,
    steps_per_execution=None 
)

reluModel.fit(encodedReviews,yTrain,batch_size=10)
pickle.dump(reluModel, open(dataPath + '/nn_relu.model.pkl', 'wb'))

predSentientTrain = (reluModel.predict(encodedReviews)).astype(int)
predSentientTest = (reluModel.predict(encodedReviewsTest)).astype(int)

print("")
print('Model Overview:')
reluModel.summary()
print("")
print('relu Model Weights & Biases:')
for layer in reluModel.layers:
    print("Layer: ", layer.name) 
    print("  --Kernels (Weights): ", layer.get_weights()[0]) 
    print("  --Biases: ", layer.get_weights()[1]) 
    
print('Test Data Predictions:')
print(classification_report(yTest, predSentientTest))
print("")

#sigmoid Creation
sigmoidModel = Sequential(name='sigmoidModel')
embeddingLayer = Embedding(4,32,input_length=86)
sigmoidModel.add(embeddingLayer)
sigmoidModel.add(Flatten())
sigmoidModel.add(Dense(128,activation='sigmoid', name='hiddenLayer'))
sigmoidModel.add(Dense(2, activation='softmax', name='outputLayer'))
L2layer = tf.keras.layers.Dense(5, kernel_initializer='ones',
                              activity_regularizer=tf.keras.regularizers.l2(0.01),
                              name='L2Layer')
sigmoidModel.add(L2layer)
sigmoidModel.add(Dropout(0.2, input_shape=(60,)))


sigmoidModel.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    loss_weights=None, 
    weighted_metrics=None, 
    run_eagerly=None,
    steps_per_execution=None 
)

sigmoidModel.fit(encodedReviews,yTrain,batch_size=10)
pickle.dump(sigmoidModel, open(dataPath + '/nn_sigmoid.model.pkl', 'wb'))

predSentientTrain = (sigmoidModel.predict(encodedReviews)).astype(int)
predSentientTest = (sigmoidModel.predict(encodedReviewsTest)).astype(int)

print("")
print('sigmoid Model Overview:')
sigmoidModel.summary()
print("")
print('Model Weights & Biases:')
for layer in sigmoidModel.layers:
    print("Layer: ", layer.name) 
    print("  --Kernels (Weights): ", layer.get_weights()[0]) 
    print("  --Biases: ", layer.get_weights()[1]) 
    
print('Test Data Predictions:')
print(classification_report(yTest, predSentientTest))
print("")

#tanh Creation
tanhModel = Sequential(name='tanhModel')
embeddingLayer = Embedding(4,32,input_length=86)
tanhModel.add(embeddingLayer)
tanhModel.add(Flatten())
tanhModel.add(Dense(128,activation='tanh', name='hiddenLayer'))
tanhModel.add(Dense(2, activation='softmax', name='outputLayer'))
L2layer = tf.keras.layers.Dense(5, kernel_initializer='ones',
                              activity_regularizer=tf.keras.regularizers.l2(0.01),
                              name='L2Layer')
tanhModel.add(L2layer)
tanhModel.add(Dropout(0.2, input_shape=(60,)))


tanhModel.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    loss_weights=None, 
    weighted_metrics=None, 
    run_eagerly=None,
    steps_per_execution=None 
)

tanhModel.fit(encodedReviews,yTrain,batch_size=10)
pickle.dump(tanhModel, open(dataPath + '/nn_tanh.model.pkl', 'wb'))

predSentientTrain = (tanhModel.predict(encodedReviews)).astype(int)
predSentientTest = (tanhModel.predict(encodedReviewsTest)).astype(int)

print("")
print('tanh Model Overview:')
tanhModel.summary()
print("")
print('Model Weights & Biases:')
for layer in tanhModel.layers:
    print("Layer: ", layer.name) 
    print("  --Kernels (Weights): ", layer.get_weights()[0]) 
    print("  --Biases: ", layer.get_weights()[1]) 
    
print('Test Data Predictions:')
print(classification_report(yTest, predSentientTest))
print("")