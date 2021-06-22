from datetime import date
import datetime
from logging import exception
from flask import Flask, redirect,url_for,request,render_template,session,flash
# from pyasn1_modules.rfc2459 import Time
# import pyrebase
import matplotlib.pyplot as plt
import numpy as np
from realtime import dl
from bowlers import dl2
from allrounder import dl3

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

app=Flask(__name__)
app.secret_key="Rudy"

@app.route('/')
def home():
    
        # else:
        #     return render_template("index.html")
    return render_template('index.html')

@app.route('/Recommended',methods=["GET","POST"])
def result():
    bow=None
    bat=None
    allrd=None
    std=None
    
    if request.method=="POST":
        std=request.form["std"]
        bow=request.form["bowlers"]
        bat=request.form["batsman"]
        allrd=request.form["allrounder"]
        batting=dl.loc[(dl["Runs"]>50.0) & (dl["Stadium"]==std)]
        print(batting)

        x=dl.iloc[:,3:16].values
        y=dl.iloc[:,-14:3].values

        x1=batting.iloc[:,3:16].values
        y1=batting.iloc[:,-14:3].values
        
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=2)
        clf.fit(X_train,y_train)
        pred2=clf.predict(x1)
        # accuracy2=metrics.accuracy_score(pred2,y_test)
        print(pred2)

        #======Bowler======#
        bowling=dl2.loc[(dl2["Wkts"]>2.0) & (dl2["Econ"]>3.0) & (dl2["Stadium"]==std)]
        bowx=dl2.iloc[:,3:15].values
        bowy=dl2.iloc[:,-12:3].values

        bowx1=bowling.iloc[:,3:15].values
        bowy1=bowling.iloc[:,-12:3].values

        bowX_train,bowX_test,bowy_train,bowy_test=train_test_split(bowx,bowy,test_size=0.3)
        from sklearn.neighbors import KNeighborsClassifier
        clf2 = KNeighborsClassifier(n_neighbors=4)
        clf2.fit(bowX_train,bowy_train)
        bowlerprediction=clf2.predict(bowx1)
        # accuracy2=metrics.accuracy_score(pred2,y_test)
        print(bowlerprediction)

        #+++++++++++++AllRounder++++++++++++#
        Allr=dl3.loc[(dl3["Bowl Av"]>25.0) & (dl3["Stadium"]==std)]

        allx=dl3.iloc[:,3:14].values
        ally=dl3.iloc[:,-10:3].values
        print(ally)

        allx1=Allr.iloc[:,3:14].values
        ally1=Allr.iloc[:,-10:3].values
        print(allx1)

        allX_train,allX_test,ally_train,ally_test=train_test_split(allx,ally,test_size=0.3)
        from sklearn.neighbors import KNeighborsClassifier
        clf3 = KNeighborsClassifier(n_neighbors=2)
        clf3.fit(allX_train,ally_train)
        allprediction=clf3.predict(allx1)
        # accuracy2=metrics.accuracy_score(pred2,y_test)
        print(allprediction)



        return render_template("result.html",batts=pred2[0:int(bat)],bowws=bowy_test[0:int(bow)],alls=allprediction[0:int(allrd)])
    else:
        return render_template('playerselection.html')
    # return render_template("playerselection.html")

if __name__=='__main__':
    app.run(debug=True)