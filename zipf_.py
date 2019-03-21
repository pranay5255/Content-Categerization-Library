import pandas as pd
import numpy as np
import gensim
import sklearn
import seaborn as sns
from preprocess_ import *
'''
#### INPUT :: 
df==input dataframe
text==text column
label_column==label column
#### OUTPUT ::
sorted dictionary with words and their frequency count in corpus
'''
def preprocess_for_filter(df,text_column='concept',label_column='label'):
    from collections import Counter
    cop_df=df.copy()
    cop_df[text_column]=cop_df[text_column].apply(lambda x: x.split(' '))
    corpus=[]
    for i in cop_df[text_column]:
        for j in i:
            corpus.append(j)
    cnt=Counter(corpus)
    sorted_dict=sorted(cnt.items(),key=lambda x: x[1],reverse=True)
    return sorted_dict
'''
#### INPUT ::
sorted_dict==sorted dictionary of all words and their frequency in corpus
alpha = rate of order (usually a power of 2) increase value in multiples of  2^n (n=subset of positive Integers) for more aggresive filtering  
#### OUTPUT ::
list of words deemed to be noisy after zipf filtering for the corpus.
'''
def zipf_filter(df,text_column='concept',label_column='label',alpha=2):
    cop_df=df.copy()
    sorted_dict=preprocess_for_filter(cop_df,text_column='concept',label_column='label')
    zipf_list=sorted_dict
    rank_list=[(i,x) for i,x in enumerate(zipf_list,1)]
    k=int(rank_list[0][1][1])/2
    alpha=2
    zipff=[(float(k)/(x[0])**alpha,np.log(x[1][1])) for x in rank_list]
    
    cnt=0
    for i in zipff:
        if i[1]!=0:
            if i[0]>i[1]:
                cnt=cnt+1
            
    
    noise_words=[x[0] for x in zipf_list[0:cnt]]
    return noise_words
    
    
    
