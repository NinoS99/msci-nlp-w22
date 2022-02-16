# -*- coding: utf-8 -*-
"""main.ipynb
"""
from __future__ import print_function
import argparse
import sys
import gensim
from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import nltk
import numpy
from nltk.corpus import stopwords
import pickle
nltk.download('stopwords')

print(sys.argv)
dataPath = sys.argv[1]

negReviews = dataPath + '/negReviews.txt'
posReviews = dataPath + '/posReviews.txt'

with open(dataPath + '/dataPath.txt', 'w') as f:
  f.write(dataPath)

#negReviews = '/content/drive/MyDrive/4B/MSCI598A1/negReviews.txt'
#posReviews = '/content/drive/MyDrive/4B/MSCI598A1/posReviews.txt'

reviews = [negReviews, posReviews]

reviewsList = []
tokenizer = RegexpTokenizer(r'\w+')
stopWords = set(stopwords.words('english'))

for file in reviews:
  with open(file) as infile:
    for line in infile:
      filteredWords = []
      review = []
      tokens = tokenizer.tokenize(line)
      #reviewsList.append(tokens)
      for t in tokens:
        if t not in stopWords:
          review.append(t)
      reviewsList.append(review)

wordToVecModel = Word2Vec(sentences=reviewsList, window=5, min_count=1, workers=4)
pickle.dump(wordToVecModel, open(dataPath + '/word2VecModel.pkl', 'wb'))
print('Word 2 Vec Model Has Been Created and Saved!')