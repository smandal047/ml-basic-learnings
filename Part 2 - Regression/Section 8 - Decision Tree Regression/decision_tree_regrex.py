import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt


FILENAME = 'Position_Salaries.csv'

dataset = pd.read_csv(FILENAME)
X = dataset.iloc[:, [1]]
y = dataset.iloc[:, [-1]]

# fitting the model
tree_regrex = DecisionTreeRegressor()
tree_regrex.fit(X, y)

# predict a new value
predict_X = 6.5
predict_y = tree_regrex.predict([[predict_X]])

# data visualization
plt.title('Decision Tree Regrex')
plt.scatter(X, y, color='red')
plt.scatter(predict_X, predict_y, color='green')
plt.plot(X, tree_regrex.predict(X))
plt.show()
