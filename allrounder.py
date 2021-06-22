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


df=pd.read_csv("datasets/sets/Allrounder_data final.csv")

print(df.shape)
print(df.columns)
print(df.head(10))
print(type(df))
print(df.dtypes)

dl3=df
# allr=df.loc[(df["Bowl Av"]>25.0) & (df["Stadium"]=='Adelaide Oval')]
# print("dfdfd")
# print(allr)


# x=df.iloc[:,3:14].values
# # print(x)
# # print(x.shape)
# y=df.iloc[:,-10:3].values
# print(y)


# X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=2)
# clf.fit(X_train,y_train)
# # ns = ['Mat','Inns','NO','Runs','HS','Ave','BF','SR','100','50','0','4s','6s'] 
# # value=np.array([2,2,0,183,92,91.5,217,84.33,0,2,0,13,0])
# # xx=pd.DataFrame(value.reshape(-1, len(value)),columns=ns)
# # # pred2=clf.predict(xx)
# Allround=clf.predict(X_test)
# accuracy2=metrics.accuracy_score(Allround,y_test)
# # print(Allround)
# # print(Allround[0:5])
# print(accuracy2)

# from sklearn import svm
# #Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel
# #Train the model using the training sets
# clf.fit(X_train, y_train)
# #Predict the response for test dataset
# predicted = clf.predict(X_test)
# print(predicted)
# # print(predicted[0:5])
# ac=metrics.accuracy_score(predicted,y_test)
# print(ac)

# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# from sklearn.ensemble import RandomForestRegressor

# regressor = RandomForestRegressor(n_estimators=10, random_state=0)
# regressor.fit(X_train, y_train)
# # ns = ['Mat','Runs','HS','Bat Av','100','Wkts','Bowl Av','Ct','St']
# # value=np.array([4,18,12,6,0,2,106.5,2,0])
# # xx=pd.DataFrame(value.reshape(-1, len(value)),columns=ns)
# y_pred = regressor.predict(X_test)

# print(y_pred)

# from sklearn import metrics

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
