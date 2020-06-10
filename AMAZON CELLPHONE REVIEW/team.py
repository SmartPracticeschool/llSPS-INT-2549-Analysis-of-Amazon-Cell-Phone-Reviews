# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:58:57 2020

@author: Dell
"""


import numpy as np
import pandas as pd
#First import the dataset in the dataset variable
#data_review variable contains the reveiws and ratings
dataset = pd.read_csv("reviews.csv")
dataset['title'].fillna(dataset['title'].mode()[0],inplace=True)
rating_final = []
review_final = []
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
for i in range(0,67986):
        review=dataset["title"][i]
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        review_final.append(review)
# the rating string contains words and numbers
# so we tokenize the numbers only from it and change into float
# for rartings below 2.5 we store the rating as poor
# for ratings between 2.5 and 3.5 the rating as average
# for ratings more than 3.5 tha rating stored as good
for loop in range(0,67986):
    rating = dataset['rating'][loop]
    rating = float(rating)
    if rating < 2.5:
        rating_final.append("poor") #poor
    elif rating >= 2.5 and rating <= 3.5 :
        rating_final.append("average") # average
    elif rating > 3.5:
        rating_final.append("good") #good
# label encode the Ratings and OneHotEncode for the classification
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
rating_final = le.fit_transform(rating_final)
rating_final = np.array(rating_final)
rating_final = np.expand_dims(rating_final, axis=1)
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
rates = one.fit_transform(rating_final).toarray()
# count vectorize the reviews according to the unique words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 20000)
x_final = cv.fit_transform(review_final).toarray()
# saving the vectorizer which would be used as dictionary.
import pickle
pickle.dump(cv, open('cv.pkl','wb'))
# Split the data into test and train sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final,rates, test_size = 0.2, random_state = 0)
# adding the neuron layers
# the units in the input layers is equal to the number of unique words
# taken three deeper layers of 2000 units each
# Relu as activation in the hidden layers
# the output layer has 3 units as the one hot encoding has 3 columns
# the classification is in categorical
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 6987, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 2000, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 2000, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 2000, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 3, init = 'random_uniform', activation = 'softmax'))
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 128,epochs = 2)
# testing the prediction
y_pred = model.predict(x_test)

text =  "Slow, annoying, fragile, heavy, and bulky... "
text = re.sub('[^a-zA-Z]', ' ',text)
text = text.lower()
text = text.split()
text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)
y_p = model.predict_classes(cv.transform([text]))
# saving the model
model.save("reviews_analysis.h5")