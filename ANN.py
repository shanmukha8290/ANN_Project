# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:36:33 2020

@author: pshan
"""
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading CSV File
dataFrame = pd.read_csv("Churn_Modelling.csv")

# Defining independent variables
X = dataFrame.iloc[:,3:13]
# Defining dependent variables
y = dataFrame.iloc[:,-1]

# creating dummies for categorical data, Gender and Geography
geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.get_dummies(X["Gender"],drop_first = True)

# Adding these data to X
X = pd.concat([X,geography,gender],axis = 1)

#Dropping the already existing columns
X = X.drop(['Geography','Gender'],axis=1)

#Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Implementing ANN

# Importing libraries required for ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

# Initialising ANN
classifier = Sequential()

# Adding Input layer and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu", input_dim = 11))

# Adding Second Hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu"))

# Adding Output Layer
classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform",activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

# Fitting ANN model to Train dataset
model_history = classifier.fit(X_train, y_train, validation_split = 0.33, epochs=100)


#predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#calculate confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

#calculate Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)















