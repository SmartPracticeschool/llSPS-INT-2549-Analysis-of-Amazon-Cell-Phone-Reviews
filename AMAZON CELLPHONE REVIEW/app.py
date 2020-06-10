# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:29:19 2019

@author: lalit
"""
    
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from keras.models import load_model
import pickle

with open(r'cv.pkl','rb') as file:
    cv=pickle.load(file)

import re 
import nltk
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
cla = load_model("reviews_analysis.h5")
cla.compile(optimizer='adam',loss='categorical_crossentropy')
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('review.html')

@app.route('/Login', methods = ['GET','POST'])
def page2():
    if request.method == 'POST':
        review = request.form['tweet']
        review = re.sub('[^a-zA-Z]', ' ',review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        review = cv.transform([review])
        
    with graph.as_default():
        y_p = cla.predict_classes(review)
    print(y_p)
    if y_p[0] == 2: 
        output = "Poor"
    elif y_p[0]== 5:
        output= "Average"
    else:
        output = "Good"
    print ("output is",output)
    return render_template('review.html',ypred = output)
        



if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
    
