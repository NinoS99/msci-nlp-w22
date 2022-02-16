# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10536N39SIziyk5-CAPaT8Y9uRYEX_goM
"""

from __future__ import print_function
import argparse
import random
import os
import sys
import nltk
import numpy

print(sys.argv)
outputFolder = sys.argv[1]
posReviews = outputFolder + '\posReviews.txt'
negReviews = outputFolder + '\\negReviews.txt'

posNegReviews = outputFolder + '\posAndNegReviews.txt'

reviews = [negReviews, posReviews]
reviewsList = []

for file in reviews:
  with open(file) as infile:
    for line in infile:
      reviewsList.append(line.strip())

random.shuffle(reviewsList) #Randomize each review for splitting it later in the code 

with open(posNegReviews, 'w') as f:
  for line in reviewsList:
    f.write(line + '\n')

#Libraries import
from nltk.corpus.reader.knbc import test
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')

#Tokenize reviews, special character removal and initialize list of stopwords
tokenizer = RegexpTokenizer(r'\w+')
stopWords = set(stopwords.words('english'))

counter = 1

with open (posNegReviews) as output, \
open(outputFolder + "/out.csv",'w') as outCSV,\
open(outputFolder + "/out_ns.csv",'w') as out_nsCSV,\
open(outputFolder + "/train_ns.csv",'w') as train_nsCSV,\
open(outputFolder + "/val_ns.csv",'w') as val_nsCSV,\
open(outputFolder + "/test_ns.csv",'w') as test_nsCSV,\
open(outputFolder + "/train.csv",'w') as trainCSV,\
open(outputFolder + "/val.csv",'w') as valCSV,\
open(outputFolder + "/test.csv",'w') as testCSV:

  for line in output:
    filteredWords = []
    tokens = tokenizer.tokenize(line)
    print(tokens, end='\n', file=outCSV)

    for t in tokens:
      if t not in stopWords:
        filteredWords.append(t)
        print(filteredWords, end='\n', file=out_nsCSV)

    trainingSet = 800000 * 0.80
    trainingCount = 0
    validationSet = (800000 * 0.10) + trainingSet
    testSet = (800000 * 0.10) + validationSet

    if counter <= trainingSet:
        print(tokens, end='\n', file=trainCSV) 
        print(filteredWords, end='\n', file=train_nsCSV) 
        counter = counter + 1
       

    elif counter > trainingSet and counter <= validationSet:
        print(tokens, end='\n', file=valCSV) 
        print(filteredWords, end='\n', file=val_nsCSV)
        counter = counter + 1
        

    else: 
        print(tokens, end='\n', file=testCSV)
        print(filteredWords, end='\n', file=test_nsCSV)
        counter = counter + 1