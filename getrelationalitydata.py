from collections import Counter
import numpy as np
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

raw_path = '/Users/Adina/Documents/Orthographic Forms/celex_valex_stats.csv'
stimlistpath = '/Users/Adina/Documents/Orthographic Forms/Stim_final.csv'

rawdata=pd.read_csv(raw_path)

df=rawdata[['target','Categories','Nfreq','Vfreq','freq','NVratio','Ncount','Nmass',
'Vditrans','Vtrans','Vintrans','Nmorph','MorphStatus']] # gives the relevant subset of the data that I need for this task

## NOUNS ##
nouns=df[df.Nfreq!=0]
nounsonly=nouns[nouns.Categories=="['noun']"] # this allows us to pull out words that are unambiguously nouns

nouns_count = nounsonly[nounsonly.Ncount==True]
nouns_count_only = nouns_count[nounsonly.Nmass==False]
nouns_count_only['rel_type']='norel'

## VERBS ##
verbs=df[df.Vfreq!=0]
verbsonly=verbs[verbs.Categories=="['verb']"] # this allows us to pull out words that are unambiguously verbs

verbs_trans = verbsonly[verbsonly.Vtrans==True]
verbs_intrans = verbsonly[verbsonly.Vintrans==True]
verbs_ditrans = verbsonly[verbsonly.Vditrans==True]

frames=[verbs_trans,verbs_ditrans]
verbs_rel = pd.concat(frames) # makes a relational verbs dataframe

verbs_intrans['rel_type']='norel'
verbs_intrans['rel_bin']=0
verbs_rel['rel_type']='rel'
verbs_rel['rel_bin']=1

frames2=[verbs_intrans,verbs_rel,nouns_count_only]
stimmies=pd.concat(frames2) # created a list of a lot of words with the codings
smallstimmies=stimmies[[u'target', u'Categories', u'Nfreq', u'Vfreq', u'freq', u'NVratio', u'Vditrans', u'Vtrans', u'Vintrans', u'Nmorph', u'MorphStatus',u'rel_type',u'rel_bin']]

### Get a list of relational nouns (i.e., ones from my studies)

stimslist=pd.read_csv(stimlistpath)
relstims=stimslist[stimslist.rel_type=='rel']
relnouns=relstims[relstims.category=='n']
relnouns['rel_bin']=1

# get the relnouns into a format where they can be easily concatenated with the verbs
relnouns=relnouns[['word','category','Nfreq','Vfreq','freq','NVratio','Vditrans','Vtrans','Vintrans','Nmorph','MorphStatus','rel_type', 'rel_bin']]
relnouns.replace(to_replace='n', value="['noun']", inplace=True, limit=None, regex=False, method='pad', axis=None)
relnouns.columns.values[1] = 'Categories'
relnouns.columns.values[0] = 'target'

framesagain=[relnouns,smallstimmies]
fulllist=pd.concat(framesagain)
fulllist.to_csv('/Users/Adina/Documents/Orthographic Forms/full_list.csv')
print 'csv of all stims is created as "full_list.csv'

frames3=[verbs_intrans,verbs_rel]
justverbs=pd.concat(frames3)
justverbs.to_csv('/Users/Adina/Documents/Orthographic Forms/justverbs.csv')
print 'csv of only verbs is created as "verbs_list.csv'

relnouns.to_csv('/Users/Adina/Documents/Orthographic Forms/relnouns.csv')
print 'csv of only relnouns is created as "relnouns_list.csv'



