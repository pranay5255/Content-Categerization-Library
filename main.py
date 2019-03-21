from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import numpy as np
import gensim
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from chi_sqaure_filter import *
from preprocess_ import * 
from zipf_ import * 
from feature_extraction import *
from models_mod_ import *
from sklearn.metrics import f1_score
import warnings
import os
import requests
import zipfile
from sklearn.preprocessing  import normalize


class preprocess_transformer(BaseEstimator,TransformerMixin):
    def __init__(self,dataset_name='other',text_column='concept',label_column='label'):
        self.dataset_name=dataset_name
        self.text_column=text_column
        self.label_column=label_column
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.dataset_name=='alexa':
            X=preprocess_for_alexa(X,text_column=self.text_column,label_column=self.label_column)
        elif self.dataset_name=='news':
            X=preprocess_for_twenty(X,text_column=self.text_column,label_column=self.label_column)
        else: 
            X = preprocess_for_gen(X,text_column=self.text_column,label_column=self.label_column)
        return X 
class filter_transformer(BaseEstimator,TransformerMixin):
    def __init__(self,alpha,low_freq_cutoff,thresh_p_value,text_column='concept',label_column='label'):
        self.alpha=alpha
        self.low_freq_cutoff=low_freq_cutoff
        self.thresh_p_value=thresh_p_value
        self.text_column=text_column
        self.label_column=label_column
    
    def fit(self,X,y):
        X=pd.concat([X,y],axis=1)
        chi_noise_words=chi_square_lowfreq_filter(X,text_column=self.text_column,\
                                                                        label_column=self.label_column,low_freq_cutoff=self.low_freq_cutoff,\
                                                  thresh_p_value=self.thresh_p_value)
        zipf_noise_words=zipf_filter(X,text_column=self.label_column,label_column=self.label_column,alpha=self.alpha)
        all_noise_words = chi_noise_words+zipf_noise_words
        self.all_noise_words=all_noise_words
        return self
    
    def transform(self,X):
        X=tokenization_with_filters(X,text_column=self.text_column,label_column=self.label_column)
        X[self.text_column]=X[self.text_column].apply(lambda x: [i for i in x if i not in self.all_noise_words])
    
        return X
    
class doc2vec_transformer(BaseEstimator,TransformerMixin):
    def __init__(self,text_column='concept',label_column='label'):        
        self.text_column=text_column
        self.label_column=label_column
    
    def transform(self,X):
        path=self.file_creator()
        base_vocab,word_vecs=read_data(path)
        vectors_tr = X[self.text_column].apply(lambda x: cword_vec(x,word_vecs,base_vocab))
        vectors_tr = normalize(vectors_tr.tolist(),axis=1)        
        return vectors_tr
    
    def fit(self,X,y=None):
        return self
    
    @staticmethod
    def file_creator():
        primary_path=os.getcwd()
        glove_path=os.path.join(primary_path,'glove')
        text_path=os.path.join(glove_path,'glove.42B.300d.txt')
        if not os.path.exists(text_path):
            glove_url='http://nlp.stanford.edu/data/glove.42B.300d.zip'
            os.makedirs(glove_path)
            global_req= requests.get(glove_url,stream=True)
            zip_file_name=os.path.join(glove_path,'glove.zip')
            with open(zip_file_name,'wb') as vector_text:
                for chunk in global_req.iter_content(chunk_size=1024):
                    if chunk:
                        vector_text.write(chunk)
            zip_ref = zipfile.ZipFile(zip_file_name, 'r')
            zip_ref.extractall(glove_path)
            zip_ref.close()
            
        return text_path
    