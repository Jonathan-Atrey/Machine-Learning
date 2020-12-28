import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:,1].values
y = data.iloc[:,2].values

X = X.reshape(-1,1)

from sklearn.linear_model import LinearRegression
linearregression_X = LinearRegression()
linearregression_X.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

plt.scatter(X,y,color = 'red')
plt.plot(X,linearregression_X.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg2.predict(X_poly),color='blue')
plt.title('Truth or Bluff (Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_pred = linearregression_X.predict([[6.5]])

poly_pred = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
