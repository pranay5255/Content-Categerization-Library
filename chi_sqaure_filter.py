import pandas as pd
import numpy as np
import gensim
import sklearn
import seaborn as sns


'''
#### INPUT :: 
word_frequency_cat=takes the word frequency per category dictionary
NOTE:can be made using function::tokens_by_category()
#### OUTPUT ::
list of all common words
'''
def common_words_forchisq(word_frequency_cat):
    ls=word_frequency_cat
    all_words=[j[0] for i in ls for j in ls[i]]
    common_words=set(all_words)
    print ("Percentage of unique words across categories = {0}".format(len(common_words)/float(len(all_words))))
    return common_words  
'''
#### INPUT: 
df==input dataframe
text==text column;
label_column==label column
#### OUTPUT:
dictionary of categories as keys with list of words and their frequency
'''
def tokens_by_category(df,text_column='concept',label_column='label',num_cols=20):
    from collections import Counter
    cop_df=df.copy()
    num_cols=num_cols
    cat_dict={}
    cat_words={}
    for i in cop_df[label_column].value_counts()[0:num_cols].index:        
        cat_dict[i]=[y for x in cop_df.loc[cop_df[label_column]==i][text_column] for y in x.split(' ')]
        cnt=Counter(cat_dict[i])
        cat_words[i]=sorted(cnt.items(),key=lambda x: x[1],reverse=True)
    return cat_words
       
def chi_square_lowfreq_filter(df,text_column='concept',label_column='label',low_freq_cutoff=5,thresh_p_value=0.3):
    from collections import defaultdict
    from scipy.stats import chisquare
    cop_df=df.copy()
    ls=tokens_by_category(cop_df,text_column='concept',label_column='label')
    comm=common_words_forchisq(ls)
    cat_freq=defaultdict(list)
    for x in ls:
        for j in ls[x]:
            if j[0] in comm:
                cat_freq[j[0]].append(j[1])
    new={x:cat_freq[x] for x in cat_freq if len(cat_freq[x])!=1}
    for x in new:
        while len(new[x])!=30:
            new[x].append(0)
    noisy=[x for x in new if chisquare(new[x])[1]>thresh_p_value and sum(new[x])>low_freq_cutoff]
    return noisy
            

    
    