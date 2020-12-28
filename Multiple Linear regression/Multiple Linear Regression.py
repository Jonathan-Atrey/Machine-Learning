
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv")

X = data.iloc[:,0:4].values
y = data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("State",OneHotEncoder(),[3])],remainder = "passthrough")
X = ct.fit_transform(X)

X = np.delete(X, 2, 1)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg = reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

acc = reg.score(X_test,y_test)






