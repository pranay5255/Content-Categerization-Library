import pandas as pd
import numpy as np
import gensim
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from chi_sqaure_filter import *
from preprocess_ import * 
from zipf_ import * 
from feature_extraction import *
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
import os
    


def eval_split(df,text_column='concept',label_column='label'):
    cop_df=df.copy()
    Y=cop_df[label_column]
    X=cop_df.drop([label_column],axis=1)
    X_train,Y_train,X_test,Y_test=train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y.values)
    return X_train,Y_train,X_test,Y_test

def random_forest_(X_train,X_test,Y_train,Y_test,features='tfidf',model_persistence=False):
    from sklearn.externals import joblib
    from sklearn.metrics import classification_report
    if features=='doc2vec':
        X_train_vec,X_test_vec=doc2vec(X_train,X_test,text_column='concept',label_column='label')

    elif features=='tfidf':
        X_train_vec,X_test_vec=tfidf(X_train,X_test,text_column='concept',label_column='label')
        
        
    clf=RandomForestClassifier(n_estimators=800,max_depth=10,min_samples_split=5,n_jobs=-1,class_weight='balanced')

    clf.fit(X_train_vec,Y_train)
    preds=clf.predict(X_test_vec)
    pred_prob=clf.predict_proba(X_test_vec)
    if model_persistence==True:
        os.makedirs('models')
        joblib.dump(clf, '../models/RF_Estim.joblib')
    
    labels = Y_test.values
    predictions = preds
        
    repo=classification_report(labels,preds)
    print (classification_report(labels,preds))
    result=classification_report_csv(repo)
    return result,pred_prob
    
        
    

def SGD_elasti_(X_train,X_test,Y_train,Y_test,features='tfidf',model_persistence=False):
    from sklearn.externals import joblib
    from sklearn.metrics import classification_report
    if features=='doc2vec':
        X_train_vec,X_test_vec=doc2vec(X_train,X_test,text_column='concept',label_column='label')

    elif features=='tfidf':
        X_train_vec,X_test_vec=tfidf(X_train,X_test,text_column='concept',label_column='label')
        
        
    clf=SGDClassifier(max_iter=1000,class_weight='balanced',penalty='elasticnet',loss='log',shuffle=True,learning_rate='optimal')

    clf.fit(X_train_vec,Y_train)
    preds=clf.predict(X_test_vec)
    pred_prob=clf.predict_proba(X_test_vec)
    
    if model_persistence==True:
        os.makedirs('models')
        joblib.dump(clf, '../models/SGD_Estim.joblib')
    
    
    labels = Y_test.values
    predictions = preds

    repo=classification_report(labels,preds)
    print (classification_report(labels,preds))
    result=classification_report_csv(repo)
    return result,pred_prob

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('    ') 
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = row_data[1]
        row['recall'] = row_data[2]
        row['f1_score'] = row_data[3]
        row['support'] = row_data[4]
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe    
    