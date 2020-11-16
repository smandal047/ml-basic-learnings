import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


FILENAME = 'Social_Network_Ads.csv'

# getting the dataset
dataset = pd.read_csv(FILENAME)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature scaling
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

# training the model
regressor = LogisticRegression(random_state=42)
regressor.fit(X_train, y_train)

# testing the model
test_y_predict = regressor.predict(X_test)
# print(list(zip(y_test, test_y_predict)))

# acc score
acc = accuracy_score(y_test, test_y_predict)
cm = confusion_matrix(y_test, test_y_predict)
print('The model scored: ', acc)
print('The confusion matrix: ', cm)

# predicting a value
predict_X = [[90, 0], [0, 0]]
predict_y = regressor.predict(X_scaler.transform(predict_X))

print('predict: \n', list(zip(predict_X, predict_y)))
