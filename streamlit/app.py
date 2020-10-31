# pip install streamlit

import streamlit as st
import pandas as pd
import xgboost
import re
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import f1_score
import pickle
from functools import partial

def clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = []
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def vectorize(type_of_vectorizer, input_text):
    out_vector = None
    if type_of_vectorizer == "unigram":
        pickle_path = "./models/tfidf_vect.pkl"
    elif type_of_vectorizer == "bigram":
        pickle_path = "./models/tfidf_vect_ngram.pkl"

    with open(pickle_path, 'rb') as file:
        pickle_model = pickle.load(file)

    out_vector = pickle_model.transform(input_text)
    return out_vector



models = {
    "Bagging_word_level": "For Unigram Tf-Idf feature vectors using Random Forest Classifier",
    "Naive_bayes_ngram": "For Bigram Tf-IDF features vectors using Naive Bayes",
    "linear_classifier_ngram": "For Bigram TF-Idf feature vectors using Logistic Regression",
    "Naive_bayes_word_level":"For Unigram Tf-IDF features vectors using Naive Bayes",
    "linear_classifier_word_level":"For Unigram TF-Idf feature vectors using Logistic Regression"
}

option2 = st.sidebar.selectbox(
    'Choose a dataset',
     ["None"], key="dataset")

option = st.sidebar.selectbox(
    'Choose a model',
     ["None"] + list(models.keys()), key = "model")

def load_return_model(model_name):
    pickle_path = "./models/"+model_name+".pkl"
    with open(pickle_path, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


if option !=  "None":
    models[option]

    

    inp_text = st.text_input("Input Text", value='', type='default')
    vtype = None
    if("ngram" in option):
            vtype = "bigram"
    else:
        vtype = "unigram"
    if(inp_text):
        
        cleaned = [clean_text(inp_text)]
        vectorized = vectorize(vtype, cleaned)
        pickle_model = load_return_model(option)
        Ypredict = pickle_model.predict(vectorized)
        Ypredict

    if st.checkbox('Predict for n random samples'):
        n_sample = st.text_input("Input Text", value='', type='default', key='n_sample')
        totalData = pd.read_csv('../nelagt/nela10.csv')
        if n_sample:
            totalData = totalData.sample(n=int(n_sample))
            totalData = totalData.drop(['id','date','source','title','author','url','published','published_utc','collection_utc'],axis=1)
            totalData.content = totalData.content.apply(clean_text)
            totalData.content = totalData.content.str.replace('\d+', '')
            typed_vectorize = partial(vectorize,vtype)
            pickle_model = load_return_model(option)
            totalData["Predicted_Reliability"] = pickle_model.predict(vectorize(vtype, totalData["content"].tolist()))
            totalData







