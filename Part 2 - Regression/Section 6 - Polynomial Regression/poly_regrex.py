import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt

FILENAME = 'Position_Salaries.csv'

dataset = pd.read_csv(FILENAME)
X = dataset.iloc[:, [1]]
y = dataset.iloc[:, [-1]]

# lin_regrex = LinearRegression()
# lin_regrex.fit(X, y)

poly_feat = PolynomialFeatures(degree=4)
X_feat = poly_feat.fit_transform(X)

poly_regrex = LinearRegression()
poly_regrex.fit(X_feat, y)

predict_X = 6.5
predict_y = sum(b*(predict_X**index) for index, b in enumerate(poly_regrex.coef_[0]))
predict_y_2 = poly_regrex.predict(poly_feat.fit_transform([[predict_X]]))
# TODO check why predict_y != predict_y_2
print(predict_X, predict_y, predict_y_2)

plt.scatter(X, y, color='red')
plt.scatter(predict_X, predict_y, color='green')
plt.scatter(predict_X, predict_y_2, color='black')
plt.plot(X, poly_regrex.predict(poly_feat.fit_transform(X)), color='blue')
plt.show()

# plt.scatter(X, y, color='red')
# plt.plot(X, lin_regrex.predict(X), color='blue')
# plt.xlabel('postion')
# plt.ylabel('salary')
# plt.show()
