import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt

FILENAME = 'Position_Salaries.csv'

# get the variables
dataset = pd.read_csv(FILENAME)
X = dataset.iloc[:, [1]]
y = dataset.iloc[:, -1]

# get the model
""" while using random forest
    > check for missing data
    > check for categorical encoding
    > NO to feature scaling
"""
random_forest_regrex = RandomForestRegressor(n_estimators=10)
random_forest_regrex.fit(X, y)

# predict value
predict_X = 6.5
predict_y = random_forest_regrex.predict([[predict_X]])

# visualization of data
plt.subplot(3, 1, 1)
plt.title('Random Forest Regression')
plt.xlabel('Positions')
plt.ylabel('Salary')
plt.scatter(X, y, color='red')
plt.plot(X, random_forest_regrex.predict(X), color='blue')
plt.scatter(predict_X, predict_y, color='green')

# visualization of data in extended points
""" A random forest has plotted a range of values
    that could best fit the value, as tree classifies the
    data as range of values
"""
plt.subplot(3, 1, 3)
X_plot = np.arange(min(X.values), max(X.values), 0.01)
X_plot = X_plot.reshape(len(X_plot), 1)
plt.title('Random Forest Regression Detailed')
plt.xlabel('Positions')
plt.ylabel('Salary')
plt.scatter(X, y, color='red')
plt.plot(X_plot, random_forest_regrex.predict(X_plot), color='blue')
plt.scatter(predict_X, predict_y, color='green')

plt.show()
