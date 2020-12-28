# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:31:11 2020

@author: jonat
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")

X = data.iloc[:,0].values
y = data.iloc[:,1].values

X=X.reshape(-1,1)
y=y.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3 )


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg = reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

acc = reg.score(X_test,y_test)          

plt.scatter(X_train, y_train, color = 'Blue')
plt.plot(X_train,reg.predict(X_train),color = 'Red')
plt.title("YEARS VS SALARY (train set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show()
   
plt.scatter(X_test, y_test, color = 'Blue')
plt.plot(X_train,reg.predict(X_train),color = 'Red')
plt.title("YEARS VS SALARY (test set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show()                                                                                                                                                    