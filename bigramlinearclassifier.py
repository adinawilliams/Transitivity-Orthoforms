from collections import Counter
import numpy as np
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#########################################
#  This script requires data that was   #
# prettified by another script called   #
# getrelationalitydata.py that makes    #
# all the data end up in the correct    # 
# format, pandas df (with at least two  #
# important colums `target' & rel_type) #               
#########################################


def runMultiNomClass(traindata, testdata, vectorizer, print_testall, print_stats, print_sel): # takes two pandas df and a bool, 
# model can be MultinomialNB() or LogisticRegression()
	
	listotrain = traindata['target'].tolist()
	trainarray = traindata.rel_type.as_matrix()
	traincounts = vectorizer.fit_transform(listotrain) #reads in test data and makes the relevant data structures

	listotest = testdata['target'].tolist()
	testarray = testdata.rel_type.as_matrix()
	testcounts = vectorizer.transform(listotest)

	clf2 = MultinomialNB().fit(traincounts,trainarray) #fitting the Multinomial classifier

	predicted = clf2.predict(testcounts)

# some optional print statements #

	if print_testall:
		for word, relthing in zip(listotest, predicted):
			print ('%r => %s' % (word, relthing))

	if print_sel:
		print 'print the values for the first 25 instances'
		print('GroundTruth:', testarray[0:25])
		print('Predicted:', predicted[0:25])

	if print_stats: 

		print 'these are the counts in each condition:'
		unique, counts = np.unique(y_test, return_counts=True)
		countsdict = dict(zip(unique, counts))
		rels = countsdict.get('rel', 'n/a').astype(float)
		norels = countsdict.get('norel', 'n/a').astype(float)
		tot= (rels+norels).astype(float)
		avgrel = np.multiply(np.divide(rels, tot),100)
		avgnorel = np.multiply(np.divide(norels, tot),100)
		print countsdict
		print '%f percent norel,' %avgnorel + ' and %f percent rel;' %avgrel +  ' %d is the total count' %(rels+norels)


		confusion = metrics.confusion_matrix(y_test, predicted)
		TP = confusion[1, 1]
		TN = confusion[0, 0]
		FP = confusion[0, 1]
		FN = confusion[1, 0]
		print '%d true positives,' %TP + ' %d true negatives,' %TN + ' %d false positives,' %FP + ' %d false negatives' %FN
		false_positive_rate = FP / float(TN + FP)
		recall = TP / float(FN + TP)
		precision = TP / float(TP + FP)
		specificity = TN / (TN + FP)
		print '%f false positives rate, ' %false_positive_rate + ' and %f is recall rate (rate of true positives)' %recall + ' and %f is precision (how precisely do we predict positives)' %precision


# some print statements #

	correct=np.mean(predicted == testarray)*100		
	print '%f percent correct' %correct

	nullacc = ((metrics.accuracy_score(y_test, predicted)) *100)
	print '%f percent null accuracy; accuracy if always predicting the most frequent class' %nullacc	

def TrainTestSplit(data, testsize, seeshape): 
#takes data, test percentage as a decimal, and bool
	y = data.rel_type.as_matrix() # y needs to be your dependent variable; i.e. what you want to predict
	trainout, testout, y_train, y_test = train_test_split(data, y, test_size=testsize)
	# test_size gives what percent of the data you want to holdout for test
	if seeshape:
		print trainout.shape, testout.shape

	print 'test and train sets created, they are called "testout" and "trainout"'

	return testout, trainout

raw_path = '/Users/Adina/Documents/Orthographic Forms/full_list.csv'
verbs_path = '/Users/Adina/Documents/Orthographic Forms/justverbs.csv'
relnouns_path =  '/Users/Adina/Documents/Orthographic Forms/relnouns.csv'


####################
#  Read in Data    #
####################

rawdata=pd.read_csv(raw_path)

count_vect = CountVectorizer()
bigram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2)) # bigram vectorizer
trigram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 3)) 
quatgram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4)) 

#fulllist.to_csv('full_list.csv')
maindata=rawdata[['target','rel_type']]
maindatabin=rawdata[['target','rel_bin']]


verbs = pd.read_csv(verbs_path)
verbies = verbs[['target','rel_type']]
verbiesbin = verbs[['target','rel_bin']]
relnouns = pd.read_csv(relnouns_path)
relnounies = relnouns[['target','rel_type']]
relnouniesbin = relnouns[['target','rel_bin']]

########################
# Running MultiNomial  #
########################

testout=[]
trainout=[]

testout, trainout = TrainTestSplit(maindata,0.2,True)

runMultiNomClass(trainout,testout,bigram_vectorizer, False, True, True)

runMultiNomClass(verbies,relnounies, bigram_vectorizer, True, True, True)


########################
# Running Logistic Reg #
########################


testout, trainout = TrainTestSplit(maindatabin,0.2,True)

runClass(trainout,testout,LogisticRegression(),bigram_vectorizer,False, True, True)

########################
# Testing with boring  #
# sets.                #
########################

 N=2988
 
# TODO for tomorrow:

# get all the numbers and put them in a table

# create a random test set that is just like, all rel or something, and see how it goes
# check to make sure there are no repeat examples, words with same stem should all be in either train or test
# try adding trigram features; should be able to memorize the data; maybe try 4-grams too; if it goes up to 98% it's broken
# check for prefixes etc.
# look at model parameters; all interpretable. 
# they are the likelihood that a rel. noun will contain every bigram, convert parameter into weight feature pair tuple for rel and non; sort by weights
# figure out how to do a confusion matrix

# print out frequency of features in both sets

# write a function that prints weights for features that fire for each example, by example
# for naive bayes if we look at feature weights on their own, it's not super informative. probability of th|rel and th|nonrel
# P(feature|rel)/P(feature|norel); will get rid of often-ness features

# maybe try logistic regression too.
# naive bayes breaks if, e.g., ther is really common, b/c whenever you see he it will be as part of th and er