from pandas import read_csv, DataFrame
from numpy import nan
# import matplotlib.pyplot as plot

""" importing data """
dataset = read_csv('Data.csv')
# print(dataset)

# data.iloc[<row selection>, <column selection>]
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
# print(x)
# print(y)


""" taking care of missing data """
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=nan, strategy='mean')

imputed_x = impute.fit(x.iloc[:, 1:])
x.iloc[:, 1:] = imputed_x.transform(x.iloc[:, 1:])
# print(x, '\n', y)

""" taking care of categorical data """
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# method 1
label_encoder = LabelEncoder()
# x.iloc[:, 0] = label_encoder.fit_transform(x.iloc[:, 0])
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
# encoded_x = one_hot_encoder.fit_transform(x.iloc[:, 0]).toarray() # not working as its returning a pd.Series
encoded_x = one_hot_encoder.fit_transform(x.iloc[:, [0]]).toarray() # working bcoz returnin pd.DataFrame
# encoded_x = one_hot_encoder.fit_transform(x[['Country']]).toarray() # working bcoz returnin pd.DataFrame

x = DataFrame(encoded_x).join(x.iloc[:, 1:])
y = label_encoder.fit_transform(y)
print(x)
print(y)

# method 2
# ct = ColumnTransformer([('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
# X = ct.fit_transform(x)

""" for test train split """
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train, '\n', y_train)

""" feature scaling """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train, '\n', x_test)

""" models """
from sklearn.linear_model import LinearRegression
regrex = LinearRegression()
regrex.fit(x_train, y_train)

y_predict = regrex.predict(x_test)
print(y_predict, '\n', y_test)