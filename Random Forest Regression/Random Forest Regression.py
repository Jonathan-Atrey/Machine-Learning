import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:,1].values
y = data.iloc[:,2].values

X = X.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
jungle = RandomForestRegressor(n_estimators=10)
jungle.fit(X,y)

prediciton = jungle.predict([[6.5]])

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,jungle.predict(X_grid),color='blue')
plt.title("Truth or Bluff (Random Forest Regression HD)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()












