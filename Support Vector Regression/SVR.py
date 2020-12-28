import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:,1].values
y = data.iloc[:,2].values

X = X.reshape(-1,1)
y = y.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
reg = SVR(kernel = 'rbf')
reg.fit(X,y)

pred = sc_y.inverse_transform(reg.predict(sc_X.transform([[6.5]])))

plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color = 'red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(reg.predict(X)),color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(reg.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title("Truth or Bluff (SVR high resolution)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



