from numpy.core.fromnumeric import std
import pandas as pd
import datetime
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# importseaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn import metrics


df=pd.read_csv("datasets/sets/Bowlers_data final.csv")

print(df.shape)
print(df.columns)
print(df.head(5))
print(type(df))
print(df.info())
# print(df.describe())
# print(df.info())
# print(df.groupby('LotShape').size())

#dropping null value columns which cross the threshold
# a=df.isnull().sum()
# print(a)
# b=a[a>(0.05*len(a))]
# df=df.drop(b.index,axis=1)
# print(df.shape)


dl2=df.dropna()
# bowling=dl2.loc[(dl2["Wkts"]>3.0) & (dl2["Econ"]>4.0) & (dl2["Stadium"]=="adileid")]
# print(bowling)
# # print(dl.shape)
print(dl2)

bowling=dl2.loc[(dl2["Wkts"]>3) & (dl2["Econ"]>3.0) & (dl2["Stadium"]=="Brisbane Cricket Ground")]
print(bowling)

x=dl2.iloc[:,3:15].values
y=dl2.iloc[:,-12:3].values

x1=bowling.iloc[:,3:15].values
y1=bowling.iloc[:,-12:3].values
# print(y)


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X_train,y_train)
# # ns = ['Mat','Inns','NO','Runs','HS','Ave','BF','SR','100','50','0','4s','6s'] 
# # value=np.array([2,2,0,183,92,91.5,217,84.33,0,2,0,13,0])
# # xx=pd.DataFrame(value.reshape(-1, len(value)),columns=ns)
# # # pred2=clf.predict(xx)
bowlerprediction=clf.predict(x1)
# accuracy2=metrics.accuracy_score(bowlerprediction,y_test)
print(bowlerprediction)
print(bowlerprediction[0:5])
# # print(accuracy2)

# # from sklearn import svm
# # #Create a svm Classifier
# # clf = svm.SVC(kernel='linear') # Linear Kernel
# # #Train the model using the training sets
# # clf.fit(X_train, y_train)
# # #Predict the response for test dataset
# # predicted = clf.predict(X_test)
# # print(predicted)
# # # print(predicted[0:5])
# # ac=metrics.accuracy_score(predicted,y_test)
# # print(ac)


