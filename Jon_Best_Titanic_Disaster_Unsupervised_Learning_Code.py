# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 16:46:58 2023

@author: JonBest
"""

# Jon Best
# 6/18/2023
# CS379 - Machine Learning
# Titanic Survivors - Unsupervised Learningc  
# The purpose of this Python code is to use the KMeans clustering algorithm to predict the number of survivors after the Titanic disaster.
 
#***************************************************************************************
# Title: Using KMeans clustering to predict survivors of the Titanic
# Author: Tracyrenee
# Date: 2021
# Availability: https://medium.com/mlearning-ai/using-kmeans-clustering-to-predict-survivors-of-the-titanic-ae3d3e959eb8
#
#***************************************************************************************

# Imported libraries include: pandas to develop dataframes, numpy to calculate complex math, 
# sklearn for machine learning functions, and matplotlib plus seasborn for graphic representation of data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
     
# Reading CSV file to retrieve the required data.
data = pd.read_excel('CS379T-Week-1-IP.xls')

# Displays dataset before changes.
print(data)

# Convert all age and fare objects to floating points
data.age = data.age.astype(float)
data.fare = data.fare.astype(float)

# Convert all null values to zero and displays results.     
data.fillna(0)
print(data.isnull().sum())
        
# Shows data information such as the number of non-null data and the object or float64 data types.
print(data.info())

# Shows Shows a description of the data in the dataframe.
print(data.describe())

# Scatter graph that shows where survivors and non-survivors are located in computer memory.
plt.figure(figsize=(25, 7))
plt.title("Non-surviving vs Surviving Comparison by Age and Fare")
ax = plt.subplot()
ax.scatter(data[data['survived'] == 1]['age'], data[data['survived'] == 1]['fare'], c='green', s=data[data['survived'] == 1]['fare'])
ax.scatter(data[data['survived'] == 0]['age'], data[data['survived'] == 0]['fare'], c='red', s=data[data['survived'] == 0]['fare'])
plt.show()
plt.close()

# Bar graph that shows the difference between Titanic surviving and non-surviving passangers.
data.groupby('survived').survived.count().plot.bar(ylim=0)
plt.title("Non-surviving vs Surviving Comparison")
plt.show()
plt.close()

# Bar graph that displays the difference between surviving females and males.
include = data[data['survived'].values == 1]
exclude = data[data['survived'].values != 1]
survived = include
survived_sex = survived.groupby('sex').survived.count()
print (survived_sex)
survived.groupby('sex').survived.count().plot.bar(ylim=0)
plt.title("Surviving Females vs Males Comparison")
plt.show()

# Violin graph that shows the age frequencies of passangers on the Titanic.
plt.figure(figsize=(10,6))
plt.title("Titanic Passanger Ages Frequency")
sns.axes_style("dark")
sns.violinplot(y=data["age"])
plt.show()

# Violin graph that displays the fare frequency of passangers on the Titanic.     
plt.figure(figsize=(10,6))
plt.title("Titanic Passanger Fare Frequency")
sns.axes_style("dark")
sns.violinplot(y=data["fare"])
plt.show()

# Convert Male and Female values to 1 and 2 respectively.
sex1={'male':1, 'female':2}
data.sex=data.sex.map(sex1)

# Ensures that all null objects have been removed.
data['sex'] = data['sex'].fillna(data['sex'].median())   
print(data.isnull().sum())

# Set X and Y values for KMeans clustering algorithm.
data["survived"].fillna(0)
y = data["survived"]
features = ["pclass", "sex", "age", "fare"]
X = data[features]

# Shows 2D graph with normalized data.
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X) 
X.shape
np.mean(X),np.std(X)
feat_cols = ['feature'+str(i) for i in range(X.shape[1])]
feat_cols
normalised = pd.DataFrame(X,columns=feat_cols)
print(normalised)

# Scatter graph that displays plotting of Kmeans clusters
from sklearn.decomposition import PCA

pca_insurance = PCA(n_components=2)
principalComponents_insurance = pca_insurance.fit_transform(X)
plt.figure(figsize=(12,8))
plt.scatter(principalComponents_insurance[:, 0], principalComponents_insurance[:, 1], c = y, alpha = 1)
plt.show()
plt.close()

# Normalizes X for KMeans clustering.
X = (X.max() - X) / (X.max() - X.min())

# KMeans clustering predictions and results
from sklearn.cluster import KMeans

kmeans = kmeans = KMeans(n_clusters=2, max_iter=500, algorithm = 'auto',random_state=1)
kmeans.fit(X)
     
correct = 0

prediction = kmeans.predict(X)

pred_df = pd.DataFrame({'actual': y, 'prediction': prediction})
print(pred_df)

# Accuracy results of comparison between actual and prediction.
for i in range(len(y)):
  if prediction[i] == y[i]:
    correct += 1

print(correct/len(y))
     


