import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

DATASET_FILENAME = "50_Startups.csv"

# read the csv file
csv_file = pd.read_csv(DATASET_FILENAME)
X = csv_file.iloc[:, :-1]
y = csv_file.iloc[:, -1]

# create dummy vars
ct = ColumnTransformer([('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = ct.fit_transform(X)

# splitting test-train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fitting into model
regrex = LinearRegression()
regrex.fit(X_train, y_train)

# predict wrt to test set
y_predict = regrex.predict(X_test)

print(regrex.coef_)
print(regrex.intercept_)

print(y_predict.astype('float'))
