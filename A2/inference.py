# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import random
import os
import sys
import nltk
import numpy
import pandas as pd
import pickle

def modelChoose(typeOfClassifier):

  if(typeOfClassifier == 'mnb_uni'):
    with open(dataPath + '/mnb_uniModel.pkl', 'rb') as f:
      classifier = pickle.load(f)
    return classifier

  elif(typeOfClassifier == 'mnb_bi'):
    with open(dataPath + '/mnb_biModel.pkl', 'rb') as f:
      classifier = pickle.load(f)
    return classifier

  elif(typeOfClassifier == "mnb_uni_bi"):
    with open(dataPath + '/mnb_uni_biModel.pkl', 'rb') as f:
      classifier = pickle.load(f)
    return classifier

  elif(typeOfClassifier == 'mnb_bi_ns'):
    with open(dataPath + '/mnb_bi_nsModel.pkl', 'rb') as f:
      classifier = pickle.load(f)
    return classifier

  elif(typeOfClassifier == 'mnb_uni_ns'):
    with open(dataPath + '/mnb_uni_nsModel.pkl', 'rb') as f:
      classifier = pickle.load(f)
    return classifier

  elif(typeOfClassifier == 'mnb_uni_bi_ns'):
    with open(dataPath + '/mnb_uni_biModel.pkl', 'rb') as f:
      classifier = pickle.load(f)
    return classifier

  else: 
    typeofClassifierInput = input('Please enter a valid classifier:')

if __name__ == '__main__':
  from main import *
  import glob
  #Accept sys arg for path to .txt file
  print(sys.argv)
  pathTxtFile = sys.argv[1]
  #Accept sys arg for type of classifier to use
  typeOfClassifierInput = sys.argv[2]
  myfile = open('data/dataPath.txt', "r")
  txt = myfile.read()
  myfile.close()
  dataPath = txt

  classifier = modelChoose(typeOfClassifierInput)


  reviewsList = []

  with open(pathTxtFile) as infile:
      for line in infile:
        reviewsList.append(line.strip())
        
  for i in range(len(reviewsList)):
    print("Sentence reviewed: " + reviewsList[i])
    prediction = classifier.predict([reviewsList[i]])
    
    if(prediction == 0):
      print("Prediction: Negative Sentient")

    if(prediction == 1):
      print("Prediction: Positive Sentient")
