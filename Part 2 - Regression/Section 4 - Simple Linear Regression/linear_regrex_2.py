# importing libs
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# splitting test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit the train model
regrex = LinearRegression()
regrex.fit(X_train, y_train)

# predicting the model wrt to test model
y_predict = regrex.predict(X_test)
print(y_predict)

# plotting the training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regrex.predict(X_train), color='blue')
plt.title('Experience vs Salary (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# plotting the test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regrex.predict(X_train), color='blue')
plt.title('Experience vs Salary (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
