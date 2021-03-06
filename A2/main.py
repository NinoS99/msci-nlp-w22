# -*- coding: utf-8 -*-

from __future__ import print_function

if __name__ == '__main__':

    import argparse
    import random
    import os
    import nltk
    import numpy
    import sys
    import pandas as pd
    import inference
    # add sys arg command line arguements here
    print(sys.argv)
    dataPath = sys.argv[1]
    
    print(dataPath)
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
    negTrainingReviewsNS = dataPath + '/train_nsNeg.csv'
    columns = ['text']
    negTrainingDfNs = pd.read_csv(negTrainingReviewsNS, delimiter = "\t", names=columns)
    negTrainingDfNs['sentient'] = 0
    negTrainingDfNs['text'] = negTrainingDfNs['text'].str.replace('\d+', '')
    negTrainingDfNs['text'] = negTrainingDfNs['text'].apply(eval).apply(' '.join)

    #Combine pos and neg reviews without stopword removal
    reviewsFrames = [posTrainingDf, negTrainingDf]
    trainReviews = pd.concat(reviewsFrames)

    #Combine pos and neg reviews WITH stopword removal
    reviewsFramesNS = [posTrainingDfNs, negTrainingDfNs]
    trainReviewsNS = pd.concat(reviewsFramesNS)

    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report
    import numpy
    import pickle

    modelPipeline = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB())])

    #mnb_uni: classifier for unigrams with stopwords
    mnb_uniModelParameters = {
        'vect__ngram_range': [(1, 1)],
        'clf__alpha': [1]
    }

    mnb_uniModel = GridSearchCV(modelPipeline, mnb_uniModelParameters, cv=10)
    mnb_uniModel.fit(trainReviews['text'], trainReviews['sentient'])

    #mnb_bi: classifier for bigrams with stopwords
    mnb_biModelParameters = {
        'vect__ngram_range': [(2, 2)],
        'clf__alpha': [1]
    }

    mnb_biModel = GridSearchCV(modelPipeline, mnb_biModelParameters, cv=10)
    mnb_biModel.fit(trainReviews['text'], trainReviews['sentient'])

    #mnb_uni_bi: classifier for bigrams and unigrams with stopwords
    mnb_uni_biModelParameters = {
        'vect__ngram_range': [(1, 2)],
        'clf__alpha': [1]
    }

    mnb_uni_biModel = GridSearchCV(modelPipeline, mnb_uni_biModelParameters, cv=10)
    mnb_uni_biModel.fit(trainReviews['text'], trainReviews['sentient'])

    #mnb_uni_ns: classifier for unigrams without stopwords
    mnb_uni_nsModelParameters = {
        'vect__ngram_range': [(1, 1)],
        'clf__alpha': [1]
    }

    mnb_uni_nsModel = GridSearchCV(modelPipeline, mnb_uni_nsModelParameters, cv=10)
    mnb_uni_nsModel.fit(trainReviewsNS['text'], trainReviewsNS['sentient'])

    #mnb_bi_ns: classifier for bigrams without stopwords
    mnb_bi_nsModelParameters = {
        'vect__ngram_range': [(2, 2)],
        'clf__alpha': [1]
    }

    mnb_bi_nsModel = GridSearchCV(modelPipeline, mnb_bi_nsModelParameters, cv=10)
    mnb_bi_nsModel.fit(trainReviewsNS['text'], trainReviewsNS['sentient'])

    #mnb_uni_bi_ns: classifier for bigrams and unigrams without stopwords
    mnb_uni_bi_nsModelParameters = {
        'vect__ngram_range': [(1, 2)],
        'clf__alpha': [1]
    }

    mnb_uni_bi_nsModel = GridSearchCV(modelPipeline, mnb_uni_bi_nsModelParameters, cv=10)
    mnb_uni_bi_nsModel.fit(trainReviewsNS['text'], trainReviewsNS['sentient'])

    pickle.dump(mnb_uniModel, open(dataPath + '/mnb_uniModel.pkl', 'wb'))
    pickle.dump(mnb_biModel, open(dataPath + '/mnb_biModel.pkl', 'wb'))
    pickle.dump(mnb_uni_biModel, open(dataPath + '/mnb_uni_biModel.pkl', 'wb'))
    pickle.dump(mnb_uni_nsModel, open(dataPath + '/mnb_uni_nsModel.pkl', 'wb'))
    pickle.dump(mnb_bi_nsModel, open(dataPath + '/mnb_bi_nsModel.pkl', 'wb'))
    pickle.dump(mnb_uni_bi_nsModel, open(dataPath + '/mnb_uni_bi_nsModel.pkl', 'wb'))


    print('Models have been created and saved!')
