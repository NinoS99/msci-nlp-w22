# -*- coding: utf-8 -*-
"""inference.ipynb
"""
from __future__ import print_function
import argparse
import random
import os
import sys
import nltk
import numpy
import pandas as pd
import pickle
from gensim.models import Word2Vec

#Accept sys arg for path to .txt file
print(sys.argv)
pathTxtFile = sys.argv[1]
print('\n')

myfile = open('data/dataPath.txt', "r")
txt = myfile.read()
myfile.close()
dataPath = txt

with open(dataPath + '/word2VecModel.pkl', 'rb') as f:
  word2VecModel = pickle.load(f)

reviewsList = []

with open(pathTxtFile) as infile:
  for line in infile:
    reviewsList.append(line.strip())
        
for i in range(len(reviewsList)):
  similarWords = word2VecModel.wv.most_similar(reviewsList[i], topn=20)

  for e in range(len(similarWords)):
    rank = e + 1
    rankNumber = str(rank)
    print('Word Reviewed: ' + reviewsList[i] + ' | Similar Word Rank ' + rankNumber + ': ')
    print(similarWords[e])
    print('\n')