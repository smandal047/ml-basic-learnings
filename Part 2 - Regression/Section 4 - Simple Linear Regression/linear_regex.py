from pandas import read_csv, DataFrame
from numpy import nan
from matplotlib.pyplot import plot, scatter, show

# importing data
dataset = read_csv('Salary_Data.csv')
# print(dataset)

# data.iloc[<row selection>, <column selection>]
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, [-1]]
# print(x)
# print(y)

# from sklearn.impute import SimpleImputer
# impute = SimpleImputer(missing_values=nan, strategy='mean')

# imputed_x = impute.fit(x.iloc[:, 1:])
# x.iloc[:, 1:] = imputed_x.transform(x.iloc[:, 1:])
# print(x, '\n', y)

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#
# label_encoder = LabelEncoder()
# # x.iloc[:, 0] = label_encoder.fit_transform(x.iloc[:, 0])
#
# one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
#
# encoded_x = one_hot_encoder.fit_transform(x.iloc[:, [0]]).toarray() # working bcoz returnin pd.DataFrame
# # encoded_x = one_hot_encoder.fit_transform(x[['Country']]).toarray() # working bcoz returnin pd.DataFrame
#
# x = DataFrame(encoded_x).join(x.iloc[:, 1:])
# # print(x)
#
# y = label_encoder.fit_transform(y)
# # print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train, '\n', y_train)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
#
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
#
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.transform(y_test)
# # print(x_train, '\n', x_test)

from sklearn.linear_model import LinearRegression

regrex = LinearRegression()
regrex.fit(x_train, y_train)

y_predict = regrex.predict(x_test)
print(y_predict, '\n', y_test)

scatter(x_train, y_train, color='red')
# plot(x_test, y_predict, color='blue')
plot(x_train, regrex.predict(x_train), color='blue')
show()
