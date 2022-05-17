# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:30:10 2022

@author: intanh
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

#import machine learning related libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

DATAFRAME = os.path.join(os.getcwd(), 'heart.csv')
SCALER_PATH = os.path.join(os.getcwd(),'scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), "model.pkl")

df = pd.read_csv(DATAFRAME)
df.head()

print("The shape of the dataset is : ", df.shape)

#To check unique value

dict = {}
for i in list(df.columns):
    dict[i] = df[i].value_counts().shape[0]

pd.DataFrame(dict,index=["unique count"]).transpose()

cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
target_col = ["output"]
print("The categorial cols are : ", cat_cols)
print("The continuous cols are : ", con_cols)
print("The target variable is :  ", target_col)

#Summarize the statistics in dataset

df[con_cols].describe().transpose()

# Check missing value

df.isnull().sum()

#EDA - Count plot for categorical data

cat_cols=['sex','exng','caa','cp','fbs','restecg','slp','thall']
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(cat_cols):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    ax = sns.countplot(x=j,data = df)

#EDA - boxplot for continuous data

con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(con_cols):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.boxplot(x=j,data = df)

#EDA for Target

ax = sns.countplot(x = "output", data=df)

#Check type

df.info()

#Correlation for continuous data

df_corr = df[con_cols].corr().transpose()
df_corr

#The correlation coefficient has values between -1 to 1
#— A value closer to 0 implies weaker correlation (exact 0 implying no correlation)
#— A value closer to 1 implies stronger positive correlation
#— A value closer to -1 implies stronger negative correlation

fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(1,1)
gs.update(wspace=0.3, hspace=0.15)
ax0 = fig.add_subplot(gs[0,0])

mask = np.triu(np.ones_like(df_corr))
ax0.text(1.5,-0.1,"Correlation Matrix",fontsize=22, fontweight='bold', fontfamily='serif', color="#000000")
df_corr = df[con_cols].corr().transpose()
sns.heatmap(df_corr,mask=mask,fmt=".1f",annot=True,cmap='YlGnBu')
plt.show()

#Pairplot to observe both distribution of single variables and relationships between two variables

sns.pairplot(df,hue='output')
plt.show()

#Conclusion from EDA

# No missing values in the data.
# There are outliers identified from boxplot in 4/5 features - which are "trtbps","chol","thalachh" and "oldpeak"
# The data consists of more than twice the number of people with sex = 1 than sex = 0 and contrast with exng where exng = 0 is twice than exng = 1 
# Low correlation between the features presented from the heatmap. The highest negative correlation between thalachh and age (-0.4) 
# From distribution plot, age don't have high correlation to get heart attack
# Low exng will get high possibilities to get heart attack. Same goes to caa.
# Medium thall shows high possibilities to get heart attack.

# Scaling
from sklearn.preprocessing import RobustScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Models
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, roc_curve

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

print('[Packages imported]')

# creating a copy of df
df1 = df

# define the columns to be encoded and scaled
cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]

# encoding the categorical columns
df1 = pd.get_dummies(df1, columns = cat_cols, drop_first = True)

# defining the features and target
X = df1.drop(['output'],axis=1)
y = df1[['output']]

# instantiating the scaler
scaler = RobustScaler()
pickle.dump(scaler, open(MMS_SAVE_PATH, 'wb'))

# scaling the continuous featuree
X[con_cols] = scaler.fit_transform(X[con_cols])

#Show first 5 rows
X.head()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print("X_train: ", X_train.shape)
print("X_test: ",X_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

#create an array of models
#to find the best model
models = []
startTime = time.time()
models.append(("LR",LogisticRegression()))

startTime = time.time()
models.append(("NB",GaussianNB()))

startTime = time.time()
models.append(("RF",RandomForestClassifier()))

startTime = time.time()
models.append(("SVC",SVC()))

startTime = time.time()
models.append(("Dtree",DecisionTreeClassifier()))

startTime = time.time()
models.append(("KNN",KNeighborsClassifier()))

#measure the accuracy 
#k-fold cross validation applied for each model to confirm unbias train and test dataset split
#k=10
for name,model in models:
    kfold = KFold(n_splits=10, random_state=22, shuffle=True)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    executionTime_m = ((time.time() - startTime))
    print('Name: ',name) 
    print(cv_result)
    print ('Avg Accuracy: ' ,np.mean(cv_result))
    result_time_m = 'Execution time: %.4f seconds\n'%executionTime_m
    print (result_time_m)

# Proposed model

# From the result, LogisticRegression achieved the best average accuracy (84.27%) with lowest execution time (o.36 sec)
# Second best result given by SVC with 82.25% average accuracy with 5 sec execution time

# instantiating the object
logreg = LogisticRegression()

# fitting the object
logreg.fit(X_train, y_train)

# calculating the probabilities
y_pred_proba = logreg.predict_proba(X_test)

# finding the predicted valued
y_pred = np.argmax(y_pred_proba,axis=1)

# printing the test accuracy
print("The test accuracy score of Logistric Regression is ", accuracy_score(y_test, y_pred))

# instantiating the object
svc = SVC()

# fitting the model
svc.fit(X_train, y_train)

# calculating the predictions
y_pred = svc.predict(X_test)

# printing the test accuracy
print("The test accuracy score of SVC is ", accuracy_score(y_test, y_pred))

#%% Model Saving
pickle.dump(classifier, open(MODEL_PATH, "wb"))