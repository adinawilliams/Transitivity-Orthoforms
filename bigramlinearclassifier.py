from collections import Counter
import numpy as np
from numpy import copy
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
from operator import itemgetter

#### Allows us to TFIDF #######

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


#########################################
#  This script requires data that was   #
# prettified by another script called   #
# getrelationalitydata.py that makes    #
# all the data end up in the correct    # 
# format, pandas df (with at least two  #
# important colums `target' & rel_type) #               
#########################################


def runClassifier(model, traindata, testdata, vectorizer, coef_dict, print_testall, print_stats, print_sel, sel_numb, print_coeffdict, listofeats, coeff_numb, tfidf): 
# takes a string corresponding to model name: 'MultiNomial' or 'Logistic'
# two pandas df
# the name of a vectorizer, binary, trinary etc.
# 'coef_dict' tells you where to write the dictionary that links features to their model coefficients
# 3 bools about whether to print test stats, or selected things
# a number for how many selected things you want to see
# a bool corresponding to printing the top X of features ('listofeats', and a number, corresponding to items the list should contain (or the string 'all'). 

	listotrain = traindata['target'].tolist() # this will be a list of all the words you fed in
	trainarray = traindata.rel_type.as_matrix() # this will be a list of codings that correspond to the words
	traincounts = vectorizer.fit_transform(listotrain) #reads in test data and makes the relevant data structures, i.e., all the bigrams, trigrams etc.
	features = vectorizer.get_feature_names()
	print ''
	print ''
	print ''
	print '#########################################'
	print(str(len(features)) + ' features created by the vectorizer')
	print('which was designated as %s') %vectorizer
	print '#########################################'


	listotest = testdata['target'].tolist()
	testarray = testdata.rel_type.as_matrix()
	testcounts = vectorizer.transform(listotest)

	if tfidf:
		tfidf_transformer = TfidfTransformer()
		traincounts = tfidf_transformer.fit_transform(traincounts)
		print "TF-IFD transform applied; traincounts overwritten"

	if model == 'MultiNomialNB':	
		print ''
		print '#########################################'
		print '#########################################'
		print '#########################################'
		print '     This is a %s' %model + ' test!'
		print '#########################################'
		print '#########################################'
		print '#########################################'
		print ''
		clf2 = MultinomialNB().fit(traincounts,trainarray) #fitting the Multinomial classifier
		score = clf2.score(traincounts, trainarray) 
		predicted = clf2.predict(testcounts)
		probs = clf2.predict_proba(testcounts)

		coeffs = clf2.coef_[0]

		for coef, feat in zip(abs(clf2.coef_[0]),features):  # should give a dictionary of features and their contribution based on coeffs
		# I also take the absolute value b/c that tells you which features contribute more; on the assumption that all features are comparable
		# which I think is well motivated based on this dataset.
			coef_dict[feat] = coef

		if coeff_numb=='all':
			listofeats.extend(list(sorted(coef_dict.iteritems(), key=itemgetter(1), reverse=True)[:]))
		else:
			listofeats.extend(list(sorted(coef_dict.iteritems(), key=itemgetter(1), reverse=True)[:coeff_numb]))


		if print_coeffdict:
			print 'your requested number of coefficients is being printed, i.e., the top %s' %coeff_numb
			print listofeats

	# some optional print statements #

		if print_testall:
			for word, relthing in zip(listotest, predicted):
				print ('%r => %s' % (word, relthing))

		if print_sel:
			print('First %s examples' %sel_numb, listotest[0:sel_numb])
			print('GroundTruth:', testarray[0:sel_numb])
			print('Predicted:', predicted[0:sel_numb])
			print('Probabilities, for test predictions', probs[0:sel_numb])
			print '#########################################'
			#print('Coeffs for features', coef_dict)
			print('number of coefficients found', len(coeffs)) 
			print('number of coefficients equals number of features? %s' %(len(coeffs)==len(features)))
			#sanity check, bigram vectorizer makes around 618 features, depending on the test train split
		print '#########################################'
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
			print '%f percent norel (null accuracy),' %avgnorel + ' and %f percent rel;' %avgrel +  ' %d is the total count used for testing' %(rels+norels)
			print '#########################################'

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
			print '%f false positives rate, ' %false_positive_rate #+ ' and %f is recall rate (rate of true positives)' %recall + ' and %f is precision (how precisely do we predict positives)' %precision
			print metrics.classification_report(y_test, predicted)
# some print statements #
		
		correct = np.mean(predicted == testarray)*100		
		correctOnTrain = score*100
		print '%f percent correct on test set' %correct + ' and %f percent correct on training set' %correctOnTrain


	if model == 'Logistic':	
		print ''
		print '#########################################'
		print '#########################################'
		print '#########################################'
		print '   This is a %s' %model + ' regression test!'
		print '#########################################'
		print '#########################################'
		print '#########################################'
		print ''


		reldict={'rel':1, 'norel':0}
		#return len(vectorizer.fit_transform(listotrain).toarray())

		clf2 = LogisticRegression().fit(traincounts,replace_with_dict(trainarray, reldict)) #fitting the Logistic classifier, and recoding
		score = clf2.score(traincounts, replace_with_dict(trainarray, reldict)) 
		predicted = clf2.predict(testcounts)
		probs = clf2.predict_proba(testcounts)


		# to plot
		# testout['rel_type'] = testout['rel_type'].map(reldict)
		# y_pred_prob = clf2.predict_proba(testcounts)[:, 0]
		# plt.rcParams['font.size'] = 12
		# plt.hist(y_pred_prob, bins=8)
		# plt.xlim(0,1)
		# plt.title('Histogram of predicted probabilities')
		# plt.xlabel('Predicted probability of non-relationality')
		# plt.ylabel('Frequency')
		# plt.show()


		coeffs = clf2.coef_[0]

		for coef, feat in zip(abs(clf2.coef_[0]),features):  # should give a dictionary of features and their contribution based on coeffs
		# I also take the absolute value b/c that tells you which features contribute more; on the assumption that all features are comparable
		# which I think is well motivated based on this dataset.
			coef_dict[feat] = coef

		if coeff_numb=='all':
			listofeats.extend(list(sorted(coef_dict.iteritems(), key=itemgetter(1), reverse=True)[:]))
		else:
			listofeats.extend(list(sorted(coef_dict.iteritems(), key=itemgetter(1), reverse=True)[:coeff_numb]))


		if print_coeffdict:
			print 'your requested number of coefficients is being printed, i.e., the top %s' %coeff_numb
			print listofeats

		if print_testall:
			for word, relthing in zip(listotest, predicted):
				print ('%r => %s' % (word, relthing))

		if print_sel:
			print('First %s examples' %sel_numb, listotest[0:sel_numb])
			print('GroundTruth:', replace_with_dict(testarray, reldict)[0:sel_numb])
			print('Predicted:', predicted[0:sel_numb])
			print('Probabilities, , for test predictions', probs[0:sel_numb])
			print '#########################################'
			# print('Coeffs for features', coeffs)
			print('number of coefficients found', len(coeffs)) # e.g., sanity check, bigram vectorizer makes 618 features
			print('number of coefficients equals number of features? %s' %(len(coeffs)==len(features)))
			print '#########################################'

		if print_stats: 

			print 'these are the counts in each condition:'
			unique, counts = np.unique(replace_with_dict(y_test, reldict), return_counts=True)
			countsdict = dict(zip(unique, counts))
			rels = countsdict.get(1, 2) # made 2 an elsewhere number here...hope that works
			norels = countsdict.get(0, 2)
			tot= (rels+norels).astype(float)
			avgrel = np.multiply(np.divide(rels, tot),100)
			avgnorel = np.multiply(np.divide(norels, tot),100)
			print countsdict
			print '%f percent norel (null accuracy),' %avgnorel + ' and %f percent rel;' %avgrel +  ' %d is the total count used for testing' %(rels+norels)
			print '#########################################'

			confusion = metrics.confusion_matrix(replace_with_dict(y_test, reldict), predicted)
			TP = confusion[1, 1]
			TN = confusion[0, 0]
			FP = confusion[0, 1]
			FN = confusion[1, 0]
			print '%d true positives,' %TP + ' %d true negatives,' %TN + ' %d false positives,' %FP + ' %d false negatives' %FN
			false_positive_rate = FP / float(TN + FP)
			recall = TP / float(FN + TP)
			precision = TP / float(TP + FP)
			specificity = TN / (TN + FP)
			print '%f false positives rate, ' %false_positive_rate #+ ' and %f is recall rate (rate of true positives)' %recall + ' and %f is precision (how precisely do we predict positives)' %precision
			print metrics.classification_report(replace_with_dict(y_test, reldict), predicted)

		correct = np.mean(predicted == replace_with_dict(testarray, reldict))*100		
		correctOnTrain = score*100
		print '%f percent correct on test set' %correct + ' and %f percent correct on training set' %correctOnTrain

		print '#########################################'



def TrainTestSplit(data, testsize, seeshape): 
#takes data, test percentage as a decimal, and bool
	y = data.rel_type.as_matrix() # y needs to be your dependent variable; i.e. what you want to predict
	trainout, testout, y_train, y_test = train_test_split(data, y, test_size=testsize)
	# test_size gives what percent of the data you want to holdout for test, assuming you feed it a float btw 0 and 1
	if seeshape:
		print trainout.shape, testout.shape
		print y_train.shape, y_test.shape

	print 'test and train sets created, they are called "testout" and "trainout"'

	return testout, trainout, y_train, y_test


def replace_with_dict(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    # searchsorted to gets the corresponding indices for a 
    # in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]

def clear_datastructures():
	testout=[]
	trainout=[]
	y_test=[]
	y_train=[]
	coef_dict_multinom = {}
	coef_dict_logistic ={}
	rankedfeats=[]
	rankedlogfeats=[]


raw_path = '/Users/Adina/Documents/Orthographic Forms/full_list.csv'
verbs_path = '/Users/Adina/Documents/Orthographic Forms/justverbs.csv'
relnouns_path =  '/Users/Adina/Documents/Orthographic Forms/relnouns.csv'



####################
#  Read in Data    #
####################

rawdata=pd.read_csv(raw_path)

count_vect = CountVectorizer()
bigram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2)) # bigram vectorizer
# bigram_vectorizer.get_feature_names()
trigram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 3)) 
quatgram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4)) 

maindata=rawdata[['target','rel_type']]

verbs = pd.read_csv(verbs_path)
verbies = verbs[['target','rel_type']]

relnouns = pd.read_csv(relnouns_path)
relnounies = relnouns[['target','rel_type']]

reldict={'rel':1, 'norel':0}

########################
# Running MultiNomial  #
########################

testout=[]
trainout=[]
y_test=[]
y_train=[]
coef_dict_multinom = {}
coef_dict_logistic ={}

rankedfeats=[]
rankedlogfeats=[]

coef_dict_logistic_bi ={}
rankedlogfeats_bi=[]

testout, trainout, y_train, y_test = TrainTestSplit(maindata,0.4,True)

print y_test

runClassifier('MultiNomialNB', trainout,testout, bigram_vectorizer, coef_dict_multinom,  False, True, True, 25, True, rankedfeats, 10, False)



########################
# Running Logistic Reg #
########################


runClassifier('Logistic', trainout, testout, trigram_vectorizer, coef_dict_logistic, False, True, True, 25, True, rankedlogfeats, 10, True)
runClassifier('Logistic', trainout, testout, bigram_vectorizer, coef_dict_logistic_bi, False, True, True, 25, True, rankedlogfeats_bi, 10, True)

#runClassifier('Logistic', verbies, relnounies, bigram_vectorizer, coef_dict_logistic, False, True, True, 25, True, rankedlogfeats, 10). 
# fix this


 
# TODO for tomorrow:

# write a function that prints weights for features that fire for each example, by example

# get all the numbers and put them in a table

# check out the tfID thing (i.e., a way to scale for frequency)

# check to make sure there are no repeat examples, words with same stem should all be in either train or test

# they are the likelihood that a rel. noun will contain every bigram, convert parameter into weight feature pair tuple for rel and non; sort by weights

# print out frequency of features in both sets

# check the data for rel nouns in the no rel noun set...

# write a function that prints weights for features that fire for each example, by example
# we need a dict that takes example string as key and value is a list/tuple of features and weights



# for naive bayes if we look at feature weights on their own, it's not super informative. probability of th|rel and th|nonrel
# P(feature|rel)/P(feature|norel); will get rid of often-ness features

# naive bayes breaks if, e.g., ther is really common, b/c whenever you see he it will be as part of th and er