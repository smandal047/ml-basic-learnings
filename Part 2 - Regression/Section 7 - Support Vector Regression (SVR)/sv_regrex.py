import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from matplotlib import pyplot as plt

FILENAME = 'Position_Salaries.csv'

# importing csv
dataset = pd.read_csv(FILENAME)
X = dataset.iloc[:, [1]]
y = dataset.iloc[:, [-1]]

# feature scaling
""" created separate scalar objects for x and y because for each feature the 
    mean and Standard Deviation is ought to be different
"""
scalar_X = StandardScaler()
scalar_y = StandardScaler()
X_scale = scalar_X.fit_transform(X)
y_scale = scalar_y.fit_transform(y)

# regression
sv_regrex = SVR(kernel='rbf')  # here kernel is the type of algo that will be used
sv_regrex.fit(X_scale, y_scale.reshape(10, ))

# predict for a const value
predict_y_value = lambda _x: scalar_y.inverse_transform(sv_regrex.predict(scalar_X.transform(_x)))
predict_x = 6.5
predict_y = predict_y_value([[predict_x]])

# visualization of model
plt.title('Using SVR')
plt.xlabel('Employee Level')
plt.ylabel('Employee Salary Range')
plt.scatter(X, y, color='red')
plt.plot(X, predict_y_value(X), color='blue')
plt.scatter(predict_x, predict_y, color='green')
plt.show()
