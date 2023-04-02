# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 01:05:54 2022

@author: basha
"""

## exercise 1.5.4 (how to cast the Iris dataset into a regression problem.)

# Start by running the exercise 1.5.3 to load the Iris data in
# "classification format":
from ex1_5_3 import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

## Classification problem
# The current variables X and y represent a classification problem, in
# which a machine learning model will use the sepal and petal dimesions
# (stored in the matrix X) to predict the class (species of Iris, stored in
# the variable y). A relevant figure for this classification problem could
# for instance be one that shows how the classes are distributed based on
# two attributes in matrix X:
X_c = X.copy();
y_c = y.copy();
attributeNames_c = attributeNames.copy();

# Consider, for instance, if it would be possible to make a single line in
# the plot to delineate any two groups? Can you draw a line between
# the Setosas and the Versicolors? The Versicolors and the Virginicas?


## Regression problem
# Since the variable we wish to predict is petal length,
# petal length cannot any longer be in the data matrix X.
# The first thing we do is store all the information we have in the
# other format in one data matrix:
data = np.concatenate((X_c, np.expand_dims(y_c,axis=1)), axis=1)
# We need to do expand_dims to y_c for the dimensions of X_c and y_c to fit.

# We know that the petal length corresponds to the third column in the data
# matrix (see attributeNames), and therefore our new y variable is:
y_r = data[:, 2]

# Similarly, our new X matrix is all the other information but without the 
# petal length (since it's now the y variable):
X_r = data[:, [0, 1, 3, 4]]

# Since the iris class information (which is now the last column in X_r) is a
# categorical variable, we will do a one-out-of-K encoding of the variable:
species = np.array(X_r[:, -1], dtype=int).T
K = species.max()+1
species_encoding = np.zeros((species.size, K))
species_encoding[np.arange(species.size), species] = 1
# The encoded information is now a 150x3 matrix. This corresponds to 150
# observations, and 3 possible species. For each observation, the matrix
# has a row, and each row has two 0s and a single 1. The placement of the 1
# specifies which of the three Iris species the observations was.

# We need to replace the last column in X (which was the not encoded
# version of the species data) with the encoded version:
X_r = np.concatenate( (X_r[:, :-1], species_encoding), axis=1) 

# Now, X is of size 150x6 corresponding to the three measurements of the
# Iris that are not the petal length as well as the three variables that
# specifies whether or not a given observations is or isn't a certain type.
# We need to update the attribute names and store the petal length name 
# as the name of the target variable for a regression:
targetName_r = attributeNames_c[2]
attributeNames_r = np.concatenate((attributeNames_c[[0, 1, 3]], classNames), 
                                  axis=0)

# Lastly, we update M, since we now have more attributes:
N,M = X_r.shape

# A relevant figure for this regression problem could
# for instance be one that shows how the target, that is the petal length,
# changes with one of the predictors in X:
i = 2  
plt.title('Iris regression problem')
plt.plot(X_r[:, i], y_r, 'o')
plt.xlabel(attributeNames_r[i]);
plt.ylabel(targetName_r);
# Consider if you see a relationship between the predictor variable on the
# x-axis (the variable from X) and the target variable on the y-axis (the
# variable y). Could you draw a straight line through the data points for
# any of the attributes (choose different i)? 
# Note that, when i is 3, 4, or 5, the x-axis is based on a binary 
# variable, in which case a scatter plot is not as such the best option for 
# visulizing the information. 



# exercise 8.1.1





# Add offset attribute
X_r = np.concatenate((np.ones((X_r.shape[0],1)),X_r),1)
attributeNames_r = [u'Offset']+attributeNames_r
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X_r,y_r):
    
    # extract training and test set for current CV fold
    X_train = X_r[train_index]
    y_train = y_r[train_index]
    X_test = X_r[test_index]
    y_test = y_r[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Petal Length in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames_r[m], np.round(w_rlr[m,-1],2)))

print('Ran Exercise 8.1.1')

