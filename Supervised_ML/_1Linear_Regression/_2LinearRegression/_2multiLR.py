import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_index=pd.read_csv("./_0multiLRdata.csv")
# df_index.drop(columns=["year","month"],axis=1,inplace=True)
# print(df_index.head())

import seaborn as sns
# sns.pairplot(df_index)
# plt.show()

# print(df_index.corr()) #correlation

# plt.scatter(df_index['interest_rate'],df_index['unemployment_rate'])
# plt.show()


## Independent And Dependent Features
X=df_index.iloc[:,  :-1]#all rows, all columns except the last
y=df_index.iloc[:, -1]#all rows, just the last column

## Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=69)

# sns.regplot(df_index['interest_rate'],df_index['unemployment_rate'])

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()

regression.fit(X_train,y_train)
print("final coeffs: ",regression.coef_)

#{cross validation (addxitional step)
from sklearn.model_selection import cross_val_score
validation_score=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=3)
print("\n",validation_score)
print(np.mean(validation_score))
#}


## prediction
y_pred=regression.predict(X_test)
print("y_pred: ",y_pred)

## perfromance metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
print("Adjusted r-squared: ",1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))


# # see this -ytest vs ypred is a linear line suggesting we have done linear regression
# plt.scatter(y_test,y_pred)
# plt.show()
# # if we normally distribute differences,
# sns.displot(y_test-y_pred,kind="kde") #kde=kernel density estimate
# plt.show()
# # ypred vs residuals(ytest-ypred)
# plt.scatter(y_pred,y_test-y_pred) #this is random tells us it is uniformly dist which is a good thing
# plt.show()

## matching with OLS
import statsmodels.api as sm
model=sm.OLS(y_train,X_train).fit()
print(model.summary()) #we can match our calculated coeffs are almost the same
