import os
import flask
import pickle
from flask import Flask, render_template, request

# General:
import pandas as pd
import numpy as np
import re
import string
from collections import  Counter
import itertools
import random
import scipy
import seaborn as sns
from string import punctuation
import unidecode


# NLTK
import nltk
from nltk import pos_tag
from nltk import tokenize
from nltk.probability import FreqDist
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet, stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer

# SCYKITLEARN
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve, cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile, chi2, f_classif 
from sklearn.decomposition import TruncatedSVD


# Plot
import matplotlib.pyplot as plt

# Imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# JobLib
from joblib import dump, load

# WordCloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Warnings
import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)
warnings.filterwarnings('ignore', category = FutureWarning)

# SciPy
from scipy.stats import randint

# Download NLTK files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('rslp')

stop_words_add = ['voce','vai','pra','de','pro','sao','vc','ta','essa', 'so', 'não', 'nao','seu','seus','sua','suas','eu','você', 'vocês','sou','somos','meu','minha','meus','minhas','teu','tua','teus','tuas','nosso','nossa','nossas','nossos', 'vosso','vossa','vossos','vossas', 'ja', 'ser', 'ai']
stop_words = nltk.corpus.stopwords.words('portuguese') + stop_words_add

app = Flask(__name__)

class SVDDimSelect(object):
    def fit(self, X, y=None):        
        try:
            self.svd_transformer = TruncatedSVD(n_components=round(X.shape[1]/2))
            self.svd_transformer.fit(X)
        
            cummulative_variance = 0.0
            k = 0
            for var in sorted(self.svd_transformer.explained_variance_ratio_)[::-1]:
                cummulative_variance += var
                if cummulative_variance >= 0.5:
                    break
                else:
                    k += 1
                
            self.svd_transformer = TruncatedSVD(n_components=k)
        except Exception as ex:
            print(ex)
            
        return self.svd_transformer.fit(X)
    
    def transform(self, X, Y=None):
        return self.svd_transformer.transform(X)
        
    def get_params(self, deep=True):
        return {}

punctuation_token = tokenize.WordPunctTokenizer()
space_token = tokenize.WhitespaceTokenizer()
list_punctuation = [point for point in punctuation]
punctuation_stopwords = list_punctuation + stop_words
without_accents = []
without_accents_stop_words = []

@app.route('/')
def index():
    return flask.render_template('index.html')
 
def tokenize(df):
    processed_sentence = list()

    for sentence in df.sentence:
        new_sentence = list()
        
        words = space_token.tokenize(sentence)
        
        for word in words:
            if word not in stop_words:
                new_sentence.append(word)
        processed_sentence.append(' '.join(new_sentence))
        
    return processed_sentence

def removePunctuation(df):
    processed_sentence = list()

    for sentence in df.sentence:
        new_sentence = list()
        
        words = punctuation_token.tokenize(sentence)
        
        for word in words:
            if word not in punctuation_stopwords:
                new_sentence.append(word)
                
        processed_sentence.append(' '.join(new_sentence))
        
    return processed_sentence

def textNormalize(df): 
    without_accents = [unidecode.unidecode(word) for word in df.sentence]
    without_accents_stop_words = [unidecode.unidecode(word) for word in punctuation_stopwords]

    df.sentence = without_accents

    processed_sentence = list()

    for sentence in df.sentence:
        new_sentence = list()
        
        words = punctuation_token.tokenize(sentence)
        
        for word in words:
            if word not in without_accents_stop_words:
                new_sentence.append(word)
        
        processed_sentence.append(' '.join(new_sentence))
        
    return processed_sentence

def toLower(df):
    processed_sentence = list()

    for sentence in df.sentence:
        new_sentence = list()
        
        sentence = sentence.lower()
        words = punctuation_token.tokenize(sentence)
        
        for word in words:
            if word not in without_accents_stop_words:
                new_sentence.append(word)
        
        processed_sentence.append(' '.join(new_sentence))
        
    return processed_sentence

def stemming(df):
    stemmer = nltk.RSLPStemmer()
    processed_sentence = list()

    for sentence in df.sentence:
        new_sentence = list()
        
        words = punctuation_token.tokenize(sentence)
        
        for word in words:
            if word not in without_accents_stop_words:
                new_sentence.append(stemmer.stem(word))
        processed_sentence.append(' '.join(new_sentence))
        
    return processed_sentence

def tfidf(df):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(df.sentence)

def preProcess(df):
    df.sentence = tokenize(df)
    df.sentence = removePunctuation(df)
    df.sentence = textNormalize(df)
    df.sentence = toLower(df)
    df.sentence = stemming(df)
    # return tfidf(df)
    return df.sentence

def predictOffensive(sample, model_name):
    model = load("{}.gz".format(model_name))
    df = pd.DataFrame([sample], columns=['sentence'])
    X = preProcess(df)
    print(X)
    return model.predict(X)

@app.route('/predict', methods = ['POST'])
def result():

    if request.method == 'POST':
        sample = request.form.to_dict()
        models = {
            "1": "logisticRegression",
            "2": "gaussianNaiveBayes",
            "3": "multinomialNB",
            "4": "randomForest"
        }
        result = predictOffensive(sample["frase"], models[sample["modelo"]])
        prediction = "Ofensivo" if result == 1 else "Não ofensivo"
        
        return render_template("predict.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)