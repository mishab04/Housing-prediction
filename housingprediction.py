# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:41:17 2018

@author: Abhinav
"""

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

os.chdir('C:/Users/Abhinav/Downloads/My Assigments/Housing Prediction/Housing_Analysis/Housing_Analysis')

housingdata = pd.read_csv('housing.csv')

housingdata.head()

housingdata.describe()

#checking the distribution of median price of house
sns.distplot(housingdata['median_house_value'])

sns.barplot(x='ocean_proximity',y = 'median_house_value',data = housingdata)

sns.barplot(x='ocean_proximity',y = 'median_house_value',data = housingdata,estimator = np.std)

sns.countplot(x = 'ocean_proximity',data = housingdata)

sns.boxplot(x='ocean_proximity',y = 'median_house_value',data = housingdata)

sns.pairplot(housingdata.iloc[:,3:10])

sns.heatmap(housingdata.iloc[:,3:10].corr(),cmap = 'coolwarm',annot = True)

data_model = housingdata.iloc[:,3:10]

data_model['total_bedrooms'].fillna((data_model['total_bedrooms'].mean()),inplace = True)

dumdata = pd.get_dummies(data_model['ocean_proximity'])

dumdata.drop(['ISLAND'],axis = 1,inplace = True)

data_model = pd.concat([data_model,dumdata],axis = 1)

data_model.drop(['ocean_proximity'],axis = 1,inplace = True)

X  = data_model.drop(['median_house_value'],axis = 1)

y = data_model['median_house_value']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)

#Building linear regression model in python

from sklearn.linear_model import LinearRegression

l_model = LinearRegression()

l_model.fit(X_train,y_train)

#Checking cofficient
l_model.coef_

#Checking R2 of train data
l_model.score(X_train, y_train)

#Checking R2 of test data

l_model.score(X_test, y_test)

## plotting residual errors in training data
plt.scatter(l_model.predict(X_train), l_model.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')

#Checking normality

from scipy import stats

stats.shapiro(l_model.predict(X_train) - y_train)

import statsmodels.stats.api as sms

from statsmodels.compat import lzip

name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(l_model.predict(X_train) - y_train)
lzip(name, test)

#rejecting null hypothesis and concluding residuals are not normally distributed

#Checking for autocorrelation

from statsmodels.stats import diagnostic as diag

diag.acorr_ljungbox(l_model.predict(X_train) - y_train,lags =1)


#we can accept null hypothesis and concluding no autocorrelation

#Checking heteroscedascity

import statsmodels.stats.api as sms

from statsmodels.compat import lzip

#GoldFeld Quant test
name = ['F statistic', 'p-value']

test = sms.het_goldfeldquandt((l_model.predict(X_train) - y_train),np.array(X_train))

lzip(name, test)

#BP test
name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
test = sms.het_breushpagan((l_model.predict(X_train) - y_train),np.array(X_train))

lzip(name, test)


#Checking multicollinarity

from statsmodels.stats.outliers_influence import variance_inflation_factor

[variance_inflation_factor(X_train.values, j) for j in range(X_train.shape[1])]





