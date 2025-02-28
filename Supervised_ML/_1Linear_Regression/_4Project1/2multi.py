## Multiple Linear Regression
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Fetch dataset
california = fetch_california_housing()  # isn't working for now

print(california)
print(california.keys())
print(california.DESCR)
print(california.target_names)
print(california.data)
print(california.target)
print(california.feature_names)

## Prepare the dataframe
dataset = pd.DataFrame(california.data, columns=california.feature_names)
print(dataset.head())

dataset['Price'] = california.target
print(dataset.head())
print(dataset.info())
print(dataset.isnull().sum())
print(dataset.describe())
print(dataset.corr())

# Correlation heatmap
sns.heatmap(dataset.corr(), annot=True)
plt.show()

print(dataset.head())

## Independent and Dependent features
X = dataset.iloc[:, :-1]  # Independent features
y = dataset.iloc[:, -1]   # Dependent features
print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Standardizing features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)

# Model Training
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Model parameters
print("Coefficients:", regression.coef_)
print("Intercept:", regression.intercept_)

# Prediction
y_pred = regression.predict(X_test)
print("True Values:", y_test)
print("Predicted Values:", y_pred)

# Performance Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

# R-squared and Adjusted R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)
adjusted_r2 = 1 - (1-r2) * (len(y_test)-1) / (len(y_test) - X_test.shape[1] - 1)
print("Adjusted R-squared:", adjusted_r2)

# Assumptions
plt.scatter(y_test, y_pred)
plt.xlabel("Test Truth Data")
plt.ylabel("Test Predicted Data")
plt.show()

residuals = y_test - y_pred
sns.displot(residuals, kind="kde")
plt.show()

plt.scatter(y_pred, residuals)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

# Pickling the model
## Pickling 
# Python pickle module is used for serialising and de-serialising a Python object structure. Any object in Python can be pickled so that it can be saved on disk. What pickle does is that it “serialises” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script
import pickle
pickle.dump(regression, open('regressor.pkl', 'wb'))
model = pickle.load(open('regressor.pkl', 'rb'))
print("Loaded Model Predictions:", model.predict(X_test))
