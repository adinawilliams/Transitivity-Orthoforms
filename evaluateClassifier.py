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
		if num==1:
			res=df
			print res
		else:	
			res=pd.merge(res, df, on='n-gram')
	res['mean']=result.mean(axis=1)
	res.sort_values(['mean'], ascending=False, inplace=True)
	res=res[['n-gram', 'mean']]

	commonsfile=pd.read_csv(path + str(gramsize)+'gramALLfeatureCountsCommons.csv')
	commons=pd.merge(res, commonsfile, on='n-gram') #gets you the common n-grams
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

df3gramcommons=pd.DataFrame()

path='/Users/Adina/git/Transitivity-Orthoforms/Results/'


for i in xrange(1,7):
	make_commons(path, i, 20)
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
