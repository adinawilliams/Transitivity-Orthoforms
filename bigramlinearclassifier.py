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

#########################################
#  This script requires data that was   #
# prettified by another script called   #
# getrelationalitydata.py that makes    #
# all the data end up in the correct    # 
# format, pandas df (with at least two  #
# important colums `target' & rel_type) #               
#########################################



def runMultiNomClass(traindata, testdata, print_testall): # takes two pandas df and a bool
	
	listotrain=traindata['target'].tolist()
	trainarray=traindata.rel_type.as_matrix()
	traincounts = bigram_vectorizer.fit_transform(listotrain) #reads in test data and makes the relevant data structures

	listotest=testdata['target'].tolist()
	testarray=testdata.rel_type.as_matrix()
	testcounts = bigram_vectorizer.transform(listotest)

	clf2=MultinomialNB().fit(traincounts,trainarray) #fitting the Multinomial classifier

	predicted= clf2.predict(testcounts)

	if print_testall:
		for word, relthing in zip(listotest, predicted):
			print ('%r => %s' % (word, relthing))

	correct=np.mean(predicted == testarray)*100		

	print '%f percent correct' %correct
	

def TrainTestSplit(data, testsize, seeshape): 
#takes data, test percentage as a decimal, and bool
	y = data.rel_type.as_matrix()# y needs to be your dependent variable; i.e. what you want to predict
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

#fulllist.to_csv('full_list.csv')
maindata=rawdata[['target','rel_type']]


verbs = pd.read_csv(verbs_path)
relnouns = pd.read_csv(relnouns_path)

########################
# Running Classifiers  #
########################

testout=[]
trainout=[]

testout, trainout = TrainTestSplit(maindata,0.2,True)

runMultiNomClass(trainout,testout,True)

runMultiNomClass(verbs,relnouns,True)


########################
# Testing with boring  #
# sets.                #
########################

 N=2988
 
# TODO for tomorrow:

# create a random test set that is just like, all rel or something, and see how it goes
# figure out how to do a confusion matrix