import pandas as pd
import numpy as np
import gensim
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from chi_sqaure_filter import *
from preprocess_ import * 
from zipf_ import * 
import cPickle as pickle




def tokenization_with_filters(df,text_column='concept',label_column='label'):
    cop_df=df.copy()
    import gensim
    from gensim.parsing.preprocessing import strip_non_alphanum
    from gensim.parsing.preprocessing import strip_numeric
    from gensim.parsing.preprocessing import strip_punctuation
    from gensim.parsing.preprocessing import remove_stopwords
    from gensim.parsing.preprocessing import strip_tags
    from gensim.parsing.preprocessing import strip_short
    from gensim.parsing.preprocessing import preprocess_string

    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags,strip_punctuation,strip_non_alphanum,strip_numeric,strip_short,remove_stopwords]
    cop_df[text_column]=cop_df[text_column].apply(lambda x: preprocess_string(x,CUSTOM_FILTERS))
    return cop_df

def remove_stopfilter_words(df,text_column='concept',label_column='label',alpha=2,low_freq_cutoff=5,thresh_p_value=0.3):
    cop_df=df.copy()
    zipf_noise_words=zipf_filter(cop_df,text_column='concept',label_column='label',alpha=2)
    chisqu_noise_words=chi_square_lowfreq_filter(cop_df,text_column='concept',label_column='label',
                                                 low_freq_cutoff=5,thresh_p_value=0.3)
    all_noise_words=zipf_noise_words+chisqu_noise_words
    cop_df=tokenization_with_filters(cop_df,text_column='concept',label_column='label')
    cop_df[text_column]=cop_df[text_column].apply(lambda x: [i for i in x if i not in all_noise_words])
    return cop_df

def tfidf(df_train,df_test,text_column='concept',label_column='label'):
    cop_df_tr=df_train.copy()
    cop_df_tr.reset_index(inplace=True)
    cop_df_te=df_test.copy()
    cop_df_te.reset_index(inplace=True)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    conc_list_tr=cop_df_tr[text_column].apply(lambda x: str(" ".join(x)))
    conc_list_te=cop_df_te[text_column].apply(lambda x: str(" ".join(x)))
    
    vectorizer=TfidfVectorizer(analyzer=str.split)
    tr_matrix=vectorizer.fit_transform(conc_list_tr)
    te_matrix=vectorizer.transform(conc_list_te)
    from sklearn.externals import joblib
    

    
    
    return tr_matrix,te_matrix

def doc2vec(df_train,df_test,text_column='concept',label_column='label'):
    path='/home/pranay/glove/glove.42B.300d.txt'
    from sklearn.preprocessing  import normalize
    cop_df_tr=df_train.copy()
    cop_df_tr.reset_index(inplace=True)
    cop_df_te=df_test.copy()
    cop_df_te.reset_index(inplace=True)
    
    base_vocab,word_vecs=read_data(path)
    vectors_tr = cop_df_tr[text_column].apply(lambda x: cword_vec(x,word_vecs,base_vocab))
    vectors_te = cop_df_te[text_column].apply(lambda x: cword_vec(x,word_vecs,base_vocab))
    
    vectors_te = normalize(vectors_te.tolist(),axis=1)
    
    vectors_tr = normalize(vectors_tr.tolist(),axis=1)
    
    return vectors_tr,vectors_te
    
    
def cword_vec(word_list,word_vec,word_vocab):
    vec_sum=np.zeros(300)
    for word in word_list:
        if word in word_vocab:
            vec_sum=vec_sum+word_vec[word]
    return vec_sum

def read_data(file_name):
    with open(file_name,'r') as f:
        word_vocab = set() 
        word2vector = {}
        for line in f:
            line_ = line.strip() 
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
    return word_vocab,word2vector
    

    
    
    
    

    