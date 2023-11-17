# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 16:46:58 2023

@author: JonBest
"""

# Jon Best
# 6/18/2023
# The purpose of this Python code is to use the Logistic Regression algorithm to determine the passengers that will survive during the Titanic disaster.
 
#***************************************************************************************
# Title: Confusion Matrix for Your Multi-Class Machine Learning Model
# Author: Mohajon, J.
# Date: 2020
# Availability: https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
#
# Title: Titanic Survival Prediction Using Machine Learning
# Author: randerson112358
# Date: 2019
# Availability: https://betterprogramming.pub/titanic-survival-prediction-using-machine-learning-4c5ff1e3fa16
#
#***************************************************************************************

# Imported libraries include: pandas to develop dataframes, numpy to calculate complex math, 
# sklearn for machine learning functions, and matplotlib plus seasborn for graphic representation of data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
     
# Reading CSV file to retrieve the required data.
titanic = pd.read_excel('CS379T-Week-1-IP.xls')

# Display the amount of survivors and non-survivors on the Titanic.
print(titanic["survived"].value_counts())

# Display the count of survivors for columns 'sex', 'age', 'sibsp', 'parch', 'fare' and 'embarked'
cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

n_rows = 2
n_cols = 3

# Shows bar graphs for the six chosen columns that compare survivors and non-survivor statistics.
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2,n_rows*3.2))

for r in range(0,n_rows):
    for c in range(0,n_cols):  
        
        i = r*n_cols+ c      
        ax = axs[r][c]
        sns.countplot(titanic[cols[i]], hue=titanic["survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="survived", loc='upper right') 
        
plt.tight_layout()
plt.show()
plt.close()

# Shows a line graph illustrates the differences between the number of males and 
# females traveling either 1st, 2nd, or 3rd class that survided on the Titanic.
print(titanic.pivot_table('survived', index='sex', columns='pclass').plot())
plt.show()
plt.close()

# Shows a bar graph that demonstrates the survival rate of each class.
print(sns.barplot(x='pclass', y='survived', data=titanic))
plt.show()
plt.close()

# Displays the survival rate by sex, age and pclass.
age = pd.cut(titanic['age'], [0, 18, 80])
print(titanic.pivot_table('survived', ['sex', age], 'pclass'))
plt.show()
plt.close()

# Ensure that dataset is void of null values.
print(titanic.isna().sum())

# Remove unneeded columns.
titanic = titanic.drop(['name', 'ticket', 'cabin', 'boat', 'home.dest'], axis=1)

#Convert object datatypes to integers for the 'sex' and 'embarked' columns.
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Transform the 'sex' column to integers.
titanic.iloc[:,2]= labelencoder.fit_transform(titanic.iloc[:,2].values)

# Transform the 'embarked' column to integers.
titanic.iloc[:,7]= labelencoder.fit_transform(titanic.iloc[:,7].values)

print(titanic['sex'].unique())
print(titanic['embarked'].unique())

# Separate the data into independent 'X' and dependent 'Y' variables
X = titanic.iloc[:, 1:8].values 
Y = titanic.iloc[:, 0].values 

# Split the dataset into 80% Training set and 20% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Use the Random Forest Classifier algorithm for the training set.
def models(X_train,Y_train):
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  print('Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  return forest

# Get training accurancy for model.
model = models(X_train,Y_train)

# Get testing accurancy for model.
dataset = titanic
X = titanic.iloc[:, 1:8].values 
y = titanic.iloc[:, 0].values 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(kernel='rbf', C=1).fit(X_train, y_train)
y_pred = svc.predict(X_test)

# Importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

# Importing accuracy_score
from sklearn.metrics import accuracy_score
print('\nRandom Forest Classifier Testing Accuracy: {:}\n'.format(accuracy_score(y_test, y_pred)))

# Showing importance of features.
forest = model[6]
importances = pd.DataFrame({'feature':titanic.iloc[:, 1:8].columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances)

# Visual display of importance
print(importances.plot.bar())
plt.show()
plt.close()

# Displays a prediction of the Random Forest Classifier model
pred = model[6].predict(X_test)
print(pred)

# Print a space
print()

# Print the actual values
print(Y_test)

# Titanic survival test - example: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
my_survival = [[3,0,45,1, 2, 100, 1]]

# Show the prediction of Random Forest Classifier model
pred = model[6].predict(my_survival)
print(pred)

if pred == 0:
  print("Oh dear! You perished with the Titanic")
else:
  print('Lovely! You survived your end.')
