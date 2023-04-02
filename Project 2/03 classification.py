# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:21:52 2022

@author: basha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot,xlabel, ylabel, show, subplot, semilogx, title, grid, legend, suptitle, tight_layout, boxplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import mcnemar


#Get Data





# exercise 1.5.1 (load the dataset as .csv file and put it in the standard format)


# Load the Iris csv data using the Pandas library
filename = '../Data/iris.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df.values  

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(0, 4)      
X = raw_data[:, cols]     #x is matrix of features

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
classLabels = raw_data[:,-1] # -1 takes the last column :y is dependent varible vector
# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels)
# We can assign each type of Iris class with a number by making a
# Python dictionary as so:
classDict = dict(zip(classNames,range(len(classNames))))
# The function zip simply "zips" togetter the classNames with an integer,
# like a zipper on a jacket. 
# For instance, you could zip a list ['A', 'B', 'C'] with ['D', 'E', 'F'] to
# get the pairs ('A','D'), ('B', 'E'), and ('C', 'F'). 
# A Python dictionary is a data object that stores pairs of a key with a value. 
# This means that when you call a dictionary with a given key, you 
# get the stored corresponding value. Try highlighting classDict and press F9.
# You'll see that the first (key, value)-pair is ('Iris-setosa', 0). 
# If you look up in the dictionary classDict with the value 'Iris-setosa', 
# you will get the value 0. Try it with classDict['Iris-setosa']

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])
# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of 
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the 
# list to a numpy array". 
# Try running this to get a feel for the operation: 
# list = [0,1,2]
# new_list = [element+10 for element in list]

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)





# Standardizes data matrix so each column has mean 0 and std 1
X = (X - np.ones((N,1))*X.mean(0))/X.std()


# Set parameters
K1 = 5
K2 = 5
lambda_interval = np.logspace(-3, 2, 20)
L = 20
L_list = np.arange(1,L+1,1)

CV1 = model_selection.KFold(n_splits = K1, shuffle = True, random_state = 1)
CV2 = model_selection.KFold(n_splits = K2, shuffle = True, random_state = 1)

error1_logistic = np.zeros((K1))
error2_logistic =  np.zeros((K2,len(lambda_interval)))
min_error_logistic = np.zeros(K1)
opt_lambda = np.zeros(K1)


error1_KNN = np.zeros((K1))
error2_KNN = np.zeros((K2,L))
x_KNN = [0] * K1

error_baseline = np.zeros((K1))

yhat = []
y_true = []
n = 0

for train_index1, test_index1 in CV1.split(X):
    X_train1 = X[train_index1,:]
    y_train1 = y[train_index1]
    X_test1 = X[test_index1,:]
    y_test1 = y[test_index1]
    
    i = 0
    for train_index2, test_index2 in CV2.split(X_train1):
        print('Crossvalidation fold: {0}/{1}'.format(n+1,i+1))
        
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[train_index2,:]
        y_test2 = y[train_index2]
        
        #Logistical Regression
        for k in range(0,len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2',multi_class='ovr', solver='liblinear', C=1/lambda_interval[k] )
            mdl.fit(X_train2, y_train2)
            y_est_log2 = mdl.predict(X_test2).T
            
            error2_logistic[i,k] = np.sum(y_est_log2 !=y_test2)/len(y_test2)
    
        #KNN
        for k in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=k);
            knclassifier.fit(X_train2, y_train2);
            
            y_est_KNN2 = knclassifier.predict(X_test2);
            error2_KNN[i,k-1] = np.sum(y_est_KNN2 != y_test2) / len(y_test2)
            
        i+=1
    
    #Logistical Regression
    min_error_logistic[n] = np.min(error2_logistic.mean(0))
    opt_lambda_idx = np.argmin(error2_logistic.mean(0))
    opt_lambda[n] = lambda_interval[opt_lambda_idx]
    
    mdl = LogisticRegression(penalty='l2',multi_class='ovr', solver='liblinear', C=1/lambda_interval[n] )
    mdl.fit(X_train1, y_train1)
    y_est_log1 = mdl.predict(X_test1).T
            
    error1_logistic[n] = np.sum(y_est_log1 !=y_test1)/len(y_test1)
    
    #KNN
    min_idx = np.argmin(error2_KNN.mean(0))
    x_KNN[n] = L_list[min_idx]
    
    knclassifier = KNeighborsClassifier(n_neighbors=x_KNN[n]);
    knclassifier.fit(X_train1, y_train1);
    y_est_KNN1 = knclassifier.predict(X_test1);
    error1_KNN[n] = np.sum(y_est_KNN1 != y_test1) / len(y_test1)
    
    #Baseline
    baseline = np.argmax(np.bincount(y_train1))
    y_est_base = np.ones((y_test1.shape[0]), dtype = int)*baseline
    error_baseline[n] = np.sum(y_est_base != y_test1) / len(y_test1)
    
    dy = []
    dy.append(y_est_base)
    dy.append(y_est_KNN1)
    dy.append(y_est_log1)
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    
    y_true.append(y_test1)
    n+=1
    

y_true = np.concatenate(y_true)
yhat = np.concatenate(yhat)


print('Errors KNN:\tErrors baseline\tErrors LOGREG')
for m in range(K1):   
    print(' ',np.round(error1_KNN[m],2),'\t\t',np.round(error_baseline[m],2),'\t\t',np.round(error1_logistic[m],2))



fig = plt.figure()
plt.plot(L_list,error2_KNN.mean(0)*100,'-o')
plt.xlabel('Number of neighbors')
plt.ylabel('Classification error rate (%)')
plt.savefig('KNN.png',dpi=300, bbox_inches='tight')

fig = plt.figure()
plt.semilogx(lambda_interval, error2_logistic.mean(0)*100,'-or')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Classification error rate (%)')
plt.savefig('Logtistic Regression.png',dpi=300, bbox_inches='tight')


fig= plt.figure()
boxes = [error1_logistic, error1_KNN,error_baseline]
boxes_df = pd.DataFrame(boxes).T
x = [1,2,3]
labels = ['Logistic Regression','KNN', 'Baseline']
plt.boxplot(boxes_df)
ylabel('Generalization Error')
plt.xticks(x,labels)
plt.savefig('boxplot_classification.png',dpi=300, bbox_inches='tight') 


alpha = 0.05

print('A : Baseline\nB : KNN')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
print('theta: ',np.round(thetahat,2),' CI: ',np.round(CI,2),' p: ',np.round(p,3))
print('\n')
print('A : Baseline\nB : Logistical Regression')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
print('theta: ',np.round(thetahat,2),' CI: ',np.round(CI,2),' p: ',np.round(p,3))
print('\n')
print('A : KNN\nB : Logistical Regression')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,1], yhat[:,2], alpha=alpha)
print('theta: ',np.round(thetahat,2),' CI: ',np.round(CI,2),' p: ',np.round(p,3))