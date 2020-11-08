import pandas as pd

FILENAME = 'Position_Salaries.csv'

# importing csv
dataset = pd.read_csv(FILENAME)
X = dataset.iloc[:, [1]]
y = dataset.iloc[:, [-1]]

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scale = scaler.fit_transform(X)
y_scale = scaler.transform(y)
print(X_scale, '\n', y_scale)