#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:22:15 2023

@author: myyntiimac
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## we build both SVR(Support vector rgression) and KNN(K-nearest neighoubour)
#Both use in regression and classification  but little differennce in import func()
# For example if dependent variable is catagorical , we use SVC(support vector classifier)
#In SVM , the algorithom make hperplane instead of regression line, and have two dicision boundary line or sV_line which 
#separate the data point by SVline, the distance between SV_line called distance matrix
#The distance matrix can be manipulate o increase to reduce the error and adjust the error. 
#in SVR, we need to consider maximum marginal distance to adjust error, to adjust outliar, so that model predict well
# Lets import data and buil model and chck performance and optimized the model by hyper_parameter tuning
df = pd.read_csv("/Users/myyntiimac/Desktop/EMP SAL.csv")
df.head()

#Now we define the dv and ind.dependent by slicing , during slicing we can also ignore position column because position and level is same identifieier

X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

#Lets develop the SVR model First
# First import SVR from sklearn.SVM and define the function
from sklearn.svm import SVR
regressor = SVR()# in SVR the default parameter, is (kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1))
#where kernel can change with linear,poly, sigmoid, precomputed. degree=4,5,6..,.gamma or kernal coefficient with auto, C=1-represnt regulirization.

#Lets train the model with default parameter

regressor.fit(X, y)

# Then predict the model with trained model at employee level 6.5

y_pred_svr = regressor.predict([[6.5]])
# we find our SVR trained model with default parameter predicted 130001, whic is low salary in this level
# so we need to build a model which predict 150k to 200k at 6.5 lvel, why?
# So that our model predict well and check the employee bluff
# so how you optimized the model ? you can do it by parameter tuning instead of default 
# Lets train the model with parameter manipulation
from sklearn.svm import SVR
regressor = SVR(kernel='sigmoid', degree=5, gamma='auto')
regressor.fit(X, y)
# check the model by prediction at level 6.5
y_pred_svr = regressor.predict([[6.5]])
#this manipulation predict 175708, which is average between 150k to 200k 
#Lets see the visulization and check how our model fit the data point 
#For this we build a scatter plot for our actual data, then build a SVR line over it to see how data fit with our SVR model prediction
# Visualising the SVR results
plt.scatter(X, y, color = 'blue')
plt.plot(X, regressor.predict(X), color = 'green')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# in our plot, we can see our predected line capture all the datapoint , but still two point is not in line , 
#Lets try a bit parameter manipulation in SVR  function
###, By examining the predicted value with newly parameter manipulated model we see model predicted 1599973 at 6.5 at degrre 5
# thats acutulay decrease the prediction, thats not better than degree 4
# By checking plot, we see that its still two point missed by predicted line .
#Lets check  anouther time with kernel change =sigmoid
#prediction at 6.5=129999-poor prediction
#Noted to use kernel precomputed , need your input be in square matrix , in here your x value is not input matrix
# for kernel=lenear, and degree=5,6, model predict 130025 and plot looking not fit with predicted line

# Lets build the KNN model
#as we define our variable so in here we just build model andpredict
#In KNN model , algorithom take dicision on the basis of majority of neighbours dicision
from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='brute')
#the default value( n_neighbors: 5,weights: 'uniform',algorithm: 'auto',leaf_size: 30,p: 2 (Euclidean distance metric),metric: 'minkowski',metric_params: None,n_jobs: None)
regressor_knn.fit(X,y)
#Predict the  trained model model at level 6.5
y_pred_knn = regressor_knn.predict([[6.5]])

# lets see how plot lok like  in default parameter
# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'blue')
plt.plot(X_grid, regressor_knn.predict(X_grid), color = 'green')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# By default parameter we can see KNN regressor not predict well and in plot visualization, predicted line can not capture data point
# Lets manipulate paramter and see hoe it perform
#






