import collections
import operator
import math
import numpy as np
import re
import spacy
from data_loader import fetch_data
from random import random
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



#A module for vectorizing the structural stats of a story
class TextStats(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text),
                 'avg_word_len': sum(len(word) for word in text)/len(text)} for text in posts]



#Run the classifier and return the output labels
def predict_text(Xt, Yt, Xv, clf, Yp):
    clf.fit(Xt, Yt)
    Yvp = clf.predict_proba(Xv)
    Yv = clf.predict(Xv)
    i = 0
    while i < len(Yv):
        if(Yv[i] == Yv[i+1] and Yv[i] == 1):
            if(Yvp[i][1] > Yvp[i+1][1]):
                Yv[i+1] = 0
            else:
                Yv[i] = 0
        elif (Yv[i] == Yv[i+1] and Yv[i] == 0):
            if(Yvp[i][0] > Yvp[i+1][0]):
                Yv[i+1] = 1
            else:
                Yv[i] = 1
        i = i+2
    acc = classification_report(Yp, Yv)
    print(acc)
    preds = []
    x = 0
    while x < len(Yv):
        if(Yv[x] == 1):
            preds.append(1)
        else:
            preds.append(2)
        x = x+2
    with open('res_nb.txt', 'w') as f:
        f.write(str(preds))

    return Yv

def predict_test(Xt, Yt, Xv, clf, Id):
    clf.fit(Xt, Yt)
    Yvp = clf.predict_proba(Xv)
    Yv = clf.predict(Xv)
    i = 0
    while i < len(Yv):
        if(Yv[i] == Yv[i+1] and Yv[i] == 1):
            if(Yvp[i][1] > Yvp[i+1][1]):
                Yv[i+1] = 0
            else:
                Yv[i] = 0
        elif (Yv[i] == Yv[i+1] and Yv[i] == 0):
            if(Yvp[i][0] > Yvp[i+1][0]):
                Yv[i+1] = 1
            else:
                Yv[i] = 1
        i = i+2
    #print("D1")
    f = open("output.csv", "w")
    f.write("Id,Prediction\n")
    x = 0
    while x < len(Yv):
        if(Yv[x] == 1):
            f.write(str(Id[x]) + ',' + str(int(1)) + '\n')
        else:
            f.write(str(Id[x]) + ',' + str(int(2)) + '\n')
        x = x+2
    return Yv

#Write the output labels to a .csv file
def write_output(labels):
    f = open("output.csv", "w")
    f.write("Id,Prediction\n")
    for i in range(labels.size):
        f.write(str(i) + ',' + str(int(labels[i])) + '\n')
    f.close()

#Preprocessor callable 
def preproccesor(document):
    doc = nlp(" ".join(document))
    d = []
    for token in doc:
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB'or token.pos_ == 'ADV'or token.pos_ == 'ADJ' or token.pos_=='NNP'):
            #d.append(token.pos_)
            d.append(token.text) 
        else:
            d.append(token.text) 
    #print(d)
    return " ".join(d)

def preproccesor1(document):
    return " ".join(document)

def vectorize(data):
    vecs = []
    labels = []
    for i, (doc, y) in enumerate(data):
    	vecs.append(doc)
    	labels.append(y)
    return vecs, labels

nlp = spacy.load("en_core_web_sm")

print("Fetching data...")
train_data, valid_data, test_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

Tx , Ty = vectorize(train_data)
Vx , Vy = vectorize(valid_data)
Tsx, Id = vectorize(test_data)

text_clf = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 5), lowercase=True,
                                     binary=False,  preprocessor=preproccesor)),
        ])),
        ('char', Pipeline([
            ('vect', CountVectorizer(analyzer='char',ngram_range=(4, 4), lowercase=False,
                                     binary=False, preprocessor=preproccesor1)),
        ])),
        ('structure', Pipeline([
            ('stats', TextStats()),
            ('DicVect', DictVectorizer())
        ]))
    ], transformer_weights={'text':1, 'structure': 1,'char':1,})),
    ('clf', LogisticRegression(n_jobs=-1,C=0.01)),
])

param_grid={
    'solver': ('lbfgs','sag','saga','liblinear'),
}
#search = GridSearchCV()
predict_test(Tx,Ty,Tsx, text_clf, Id)
print('Done ')
acc = predict_text(Tx, Ty, Vx, text_clf, Vy)