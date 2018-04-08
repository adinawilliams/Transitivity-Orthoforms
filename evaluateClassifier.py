from collections import Counter
import numpy as np
from numpy import copy
import random, re, csv, glob, os
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


def get_topn(path, gramsize, n_int):  #n-int is the top and bottom n, gramsize-gram examples that you want to see
	res=[]
	for num in xrange(1,6):
		name= 'logistic'+ str(gramsize) + 'gramRun' + str(num) + 'coefs'
		df=pd.read_csv(path+name)
		df.columns=['n-gram','weight']
		print df
		if num==1:
			res=df
			print res
		else:	
			res=pd.merge(res, df, on='n-gram')
	res['mean']=res.mean(axis=1)
	print res
	res.sort_values(['mean'], ascending=False, inplace=True)
	res=res[['n-gram', 'mean']]

	commonsfile=pd.read_csv(path + str(gramsize)+'gramALLfeatureCountsCommons.csv')
	commons=pd.merge(res, commonsfile, on='n-gram') #gets you the common n-grams
	commons.dropna(inplace=True)
	heads=commons.head(n_int)
	tails=commons.tail(n_int)
	tails.sort_values(['mean'], ascending=True, inplace=True)
	ngramcommons=pd.concat([heads,tails])
	ngramcommons=ngramcommons[['n-gram', 'mean', 'count']]
	return ngramcommons

def make_commons(path, gramsize, n_int): #n_int is the fewest number of instances of the n-gram you want, this is how we subset the dataset to common ones
	ddff=pd.read_csv(path+ str(gramsize)+'gramALLfeatureCounts', header=-1, names=['n-gram','count'])
	ddff=ddff[ddff>n_int]
	ddff.dropna(inplace=True)
	print ddff.describe()
	ddff.to_csv(path+str(gramsize)+'gramALLfeatureCountsCommons.csv')
	print len(ddff)


def get_right_wrong(path, testname, goldlab_filename, print_true):
	fin=[]
	cert=[]
	pred=[]
	df=pd.read_csv(path+testname +'.csv', header=-1)
	df.columns=['target','norelPred', 'relPred']
	goldlab=pd.read_csv(path+goldlab_filename)
	fin=pd.merge(goldlab,df, on='target')
	fin=fin[['target', 'rel_type','relPred']]
	for i in fin.relPred:
		cert.append(abs(float(0.5-i)))
		if i>0.5:
			pred.append('rel')
		else:
			pred.append('norel')
	fin['certainty']=cert
	fin['prediction']=pred
	fin=fin[['target','rel_type','certainty','prediction']]
	fin['rightwrong']=np.where(fin.rel_type==fin.prediction, True, False)
	corrects=fin[fin['rightwrong']==True]
	wrongs=fin[fin['rightwrong']==False]
	corrects.sort_values(['certainty'],ascending=False, inplace=True)
	print'these are the ones the model got right; it got %d correct' %len(corrects)
	print "here's how many gold label of each (correct)"
	print corrects.rel_type.value_counts()
	print "the model guesses correct with this average certainty (.5 max, 0 min)"
	print corrects.certainty.mean()
	corrects.to_csv(path+testname+'CorrectExs.csv')
	wrongs.sort_values(['certainty'],ascending=False, inplace=True)
	print 'these are the ones the model got wrong; it got %d wrong' %len(wrongs)
	print "here's how many gold label of each (wrong)"
	print wrongs.rel_type.value_counts()
	print "the model guesses incorrectly with this average certainty (.5 max, 0 min)" 
	print wrongs.certainty.mean()
	wrongs.to_csv(path+testname+'WrongExs.csv')

	if print_true:
		print corrects, wrongs





df3gramcommons=pd.DataFrame()

path='/Users/Adina/git/Transitivity-Orthoforms/Results/'



for i in xrange(1,7):
	make_commons(path, i, 15)
	get topn(path, i, 15)


# make_commons(path, 1, 20)
# get_topn(path, 1, 15)

# make_commons(path,2, 20)
# get_topn(path, 2, 15)

# make_commons(path,3, 20)
# get_topn(path, 3, 15)

# make_commons(path,4, 20)
# get_topn(path, 4, 15)

# make_commons(path,5, 20)
# get_topn(path, 5, 15)

# make_commons(path,6, 20)
# get_topn(path, 6, 15)
get_topn(path, 1, 15)
get_topn(path, 2, 15)
get_topn(path, 3, 15)
get_topn(path, 4, 15)
get_topn(path, 5, 15)
get_topn(path, 6, 15)


probs6= pd.read_csv(path+'6gramALLRunsProbs.csv',header=-1)
probs6.columns=['n-gram','PNoRel','PRel']
probs5= pd.read_csv(path+'5gramALLRunsProbs.csv',header=-1)
probs5.columns=['n-gram','PNoRel','PRel']
probs4= pd.read_csv(path+'4gramALLRunsProbs.csv',header=-1)
probs4.columns=['n-gram','PNoRel','PRel']
probs3= pd.read_csv(path+'3gramALLRunsProbs.csv',header=-1)
probs3.columns=['n-gram','PNoRel','PRel']
probs2= pd.read_csv(path+'2gramALLRunsProbs.csv',header=-1)
probs2.columns=['n-gram','PNoRel','PRel']
probs1= pd.read_csv(path+'1gramALLRunsProbs.csv',header=-1)
probs1.columns=['n-gram','PNoRel','PRel']

probslist=[probs6,probs5,probs4,probs3,probs2,probs1]

county=0
for j in probslist:
	county+=1
	j.sort_values(['PRel'], ascending=False, inplace=True)
	heads=j.head(20)
	tails=j.tail(20)
	wordslist=pd.concat([heads,tails])
	print 'wordslist for %s grams' %county
	print wordslist

featwe1=pd.read_csv(path+'logistic1gramRun5featweights')
featwe2=pd.read_csv(path+'logistic2gramRun5featweights')
print '2gram feat weights loaded'
featwe3=pd.read_csv(path+'logistic3gramRun5featweights')
print '3gram feat weights loaded'
featwe4=pd.read_csv(path+'logistic4gramRun5featweights')
print '4gram feat weights loaded'
featwe5=pd.read_csv(path+'logistic5gramRun5featweights')
print '5gram feat weights loaded'
featwe6=pd.read_csv(path+'logistic6gramRun5featweights')
print '6gram feat weights loaded'
print 'all feat weights loaded'
