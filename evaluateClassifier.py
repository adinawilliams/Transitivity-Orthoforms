from collections import Counter
import numpy as np
from numpy import copy
import random, re, csv, glob, os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

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
	entropy=[]
	df=pd.read_csv(path+testname +'.csv', header=-1)
	df.columns=['target','norelPred', 'relPred']
	goldlab=pd.read_csv(path+goldlab_filename)
	fin=pd.merge(goldlab,df, on='target')
	fin=fin[['target', 'rel_type','relPred','norelPred']]
	for i in fin.relPred:
		cert.append(abs(float(0.5-i)))
		if i>0.5:
			pred.append('rel')
		else:
			pred.append('norel')

	dfPredsOnly=fin[['relPred','norelPred']]
	listoprobs= dfPredsOnly.values.tolist()
	for j in range(len(listoprobs)):
		entrop=scipy.stats.entropy(listoprobs[j], base=2)
		entropy.append(entrop)
	fin['entropy']=entropy
	fin['certainty']=cert
	fin['prediction']=pred
	fin=fin[['target','rel_type','certainty','prediction','entropy']]
	fin['rightwrong']=np.where(fin.rel_type==fin.prediction, True, False)
	corrects=fin[fin['rightwrong']==True]
	wrongs=fin[fin['rightwrong']==False]
	corrects.sort_values(['certainty'],ascending=False, inplace=True)

	print ''
	print '#########################'
	print '###### Correct ##########'
	print '#########################'
	print ''
	print'these are the ones the model got right; it got %d correct' %len(corrects)
	print "here's how many gold label of each (correct)"
	print corrects.rel_type.value_counts()
	correctrels=corrects[corrects['rel_type']=='rel']
	correctnorels=corrects[corrects['rel_type']=='norel']

	print "the model guesses correct with this average certainty (.5 max, 0 min): %f" %corrects.certainty.mean()
	print "the model guesses correct with this standard deviation on certainty (.5 max, 0 min): %f" %corrects.certainty.std()
	print ''
	print "the model guesses gold labeled relational words correct with this average certainty (.5 max, 0 min): %f" %correctrels.certainty.mean()
	print "the model guesses gold labeled relational words correct with this standard deviation on certainty (.5 max, 0 min): %f" %correctrels.certainty.std()
	print ''
	print "the model guesses gold labeled non-relational words correct with this average certainty (.5 max, 0 min): %f" %correctnorels.certainty.mean()
	print "the model guesses gold labeled non-relational words correct with this standard deviation on certainty (.5 max, 0 min): %f" %correctnorels.certainty.std()
	print ''
	print "the model guesses correct with this average entropy: %f" %corrects.entropy.mean()
	print "the model guesses correct with this standard deviation on entropy: %f" %corrects.entropy.std()
	print ''
	print "the model guesses gold labeled relational words correct with this average entropy: %f" %correctrels.entropy.mean()
	print "the model guesses gold labeled relational words correct with this standard deviation on entropy: %f" %correctrels.entropy.std()
	print ''
	print "the model guesses gold labeled non-relational words correct with this average entropy: %f" %correctnorels.entropy.mean()
	print "the model guesses gold labeled non-relational words correct with this standard deiation on entropy: %f" %correctnorels.entropy.std()
	corrects.to_csv(path+testname+'CorrectExs.csv')

	print ''
	print '#########################'
	print '###### Incorrect ########'
	print '#########################'
	print ''
	wrongs.sort_values(['certainty'],ascending=False, inplace=True)
	print 'these are the ones the model got wrong; it got %d wrong' %len(wrongs)
	print "here's how many gold label of each (wrong)"
	print wrongs.rel_type.value_counts()
	wrongrels=wrongs[wrongs['rel_type']=='rel']
	wrongnorels=wrongs[wrongs['rel_type']=='norel']

	print "the model guesses incorrectly with this average certainty (.5 max, 0 min): %f" %wrongs.certainty.mean()
	print "the model guesses incorrectly with this standard deviation on certainty (.5 max, 0 min): %f" %wrongs.certainty.std()
	print ''
	print "the model guesses gold labeled relational words incorrectly with this average certainty (.5 max, 0 min): %f" %wrongrels.certainty.mean()
	print "the model guesses gold labeled relational words incorrectly with this standard deviation on certainty (.5 max, 0 min): %f" %wrongrels.certainty.std()
	print ''
	print "the model guesses gold labeled non-relational words incorrectly with this average certainty (.5 max, 0 min): %f" %wrongnorels.certainty.mean()
	print "the model guesses gold labeled non-relational words incorrectly with this standard deviation on certainty (.5 max, 0 min): %f" %wrongnorels.certainty.std()
	print ''
	print "the model guesses incorrectly with this average entropy: %f" %wrongs.entropy.mean()
	print "the model guesses incorrectly with this standard deviation on entropy: %f" %wrongs.entropy.std()
	print ''
	print "the model guesses gold labeled relational words incorrectly with this average entropy: %f" %wrongrels.entropy.mean()
	print "the model guesses gold labeled relational words incorrectly with this standard deviation on entropy: %f" %wrongrels.entropy.std()
	print ''
	print "the model guesses gold labeled non-relational words incorrectly with this average entropy: %f" %wrongnorels.entropy.mean()
	print "the model guesses gold labeled non-relational words incorrectly with this standard deiation on entropy: %f" %wrongnorels.entropy.std()

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



 def perfOps(A):
 	m=len(A)
 	n=len(A[0])
 	B=[]
 	for i in xrange(len(A)):
 		B.append([0]*n)
 		for j in xrange(len(A[i])):
 			B[i][n-1-j]=A[i][j]
 			print n-1-j
 	return B


