import pandas as pd
import numpy as np
import gensim
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns


def preprocess_for_alexa(df,text_column='concept',label_column='label'):
    cop_df=df.copy()
    cop_df.drop([x for x in cop_df.columns if x not in ['concept','label','document']],axis=1,inplace=True)
    cop_df[text_column]=cop_df[text_column].apply(lambda x: [i for i in str(x).split(',') if ' ' not in i])
    cop_df[text_column]=cop_df[text_column].apply(lambda x: ' '.join(x))
    return cop_df

def preprocess_for_twenty(df,text_column='concept',label_column='label'):
    import re
    cop_df=df.copy()
    cop_df.drop([x for x in cop_df.columns if x not in ['concept','label','document']],axis=1,inplace=True)
    cop_df[text_column]=cop_df[text_column].apply(lambda x: re.sub(r'(Newsgroups: )\w+.\w+','',x))
    return cop_df

def preprocess_for_gen(df,text_column='concept',label_column='label'):
    cop_df=df.copy()
    cop_df.drop([x for x in cop_df.columns if x not in ['concept','label','document']],axis=1,inplace=True)
    cop_df[text_column]=cop_df[text_column].apply(lambda x: x.decode('ascii','ignore'))
    return cop_df
    