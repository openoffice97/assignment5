# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 02:26:08 2020

@author: Touqeer Malik
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 00:50:49 2020

@author: Touqeer Malik
"""

#Part four

#importing libraries
import sys
import os
import pathlib 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math
import random
%matplotlib inline




#Part four - step 1 - load the dataset ----> run only

#Load the dataset
dataset=pd.read_csv('house_pricing_data_2attrs.csv')

#Part four - step 2 - split dataframe to X (feature vectors) and y (classes) ----> run only
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,2].values

plt.scatter(x,y)
plt.show()

#Part four - step 3 - split the dataset to a train-set and a test-set ----> run only
#ratio is 80:20 --> 80% data for training and 20% Data for testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2, random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train) 
x_test=sc_x.fit_transform(x_test)





#multivariate linear regression with gradient descent

m = len(y)
x_quad = [n/10 for n in range(0, 100)]
y_quad = [(n-4)**2+5 for n in x_quad]
iterations = 1500
alpha = 0.01
theta = np.array([0, 0])

def cost_function(x, y, theta):
    """
    cost_function(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    m = len(y) 
    
    ## Calculate the cost with the given parameters
    J = np.sum((x.dot(theta)-y)**2)/2/m
    
    return J
cost_function(x, y, theta)

def gradient_descent(X, y, theta, alpha, iterations):
    """
    gradient_descent Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    cost_history = [0] * iterations
    
    for iteration in range(iterations):
        hypothesis = x.dot(theta)
        loss = hypothesis-y
        gradient = x.T.dot(loss)/m
        theta = theta - alpha*gradient
        cost = cost_function(x, y, theta)
        cost_history[iteration] = cost

    return theta, cost_history
(t, c) = gradient_descent(x,y,theta,alpha, iterations)
print t


## Plotting the best fit line
best_fit_x = np.linspace(0, 25, 20)
best_fit_y = [t[1] + t[0]*xx for xx in best_fit_x]



#fitting linear regression model to the training set.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)
#prediction of test set
y_pred=regressor.predict(x_test)

































