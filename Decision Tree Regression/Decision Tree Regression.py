import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sc

data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:,1].values
y = data.iloc[:,2].values

X = X.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor
pedh = DecisionTreeRegressor()
pedh.fit(X,y)

pred_val = pedh.predict([[6.5]])

X_grid =  np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = "red")
plt.plot(X_grid,pedh.predict(X_grid),color="blue")
plt.show()
