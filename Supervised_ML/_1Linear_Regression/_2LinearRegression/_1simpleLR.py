#simple because only one feature
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("./_0simpleLRdata.csv")

## Independent and dependent features
X=df[['weight']] #to make np.array(X).shape=(a,1)
#independent features should be data frame or 2D array
y=df['height'] #shape would be (a,)
#dependent feature can be in series or 1D array

## TRAIN & TEST split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
                                                #defining random_state always randomize samely.
## Standardization using zscore
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
# fit transform use kiya toh x_test ka sigma aur mu calc kar lega aur hum nahi chahte
        #kyuki hum pretend karenge hume kuchh nahi pata test data ke bare me to prevent
        #DATA LEAKAGE. isliye sirf transform use karenge toh wo X_train ka calculated
        #mu aur sigma hi X_test pe laga dega
X_test=scaler.transform(X_test)

## Apply Simple Linear Regression
from sklearn.linear_model import LinearRegression
#we made xtrain a 2d arr bcz here the lr function doesnt take 1d arr for features
regression=LinearRegression()
regression.fit(X_train,y_train)
print("Slope or coeff of best fitted line: ",regression.coef_)#13.11131611
print("Intercept: ",regression.intercept_)                    #161.44597222222222

## Plotting the best fit line
plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train))
# plt.show()

## Prediction for test data
y_pred=regression.predict(X_test)
## Prediction for new data
prediction = regression.predict(scaler.transform(pd.DataFrame([[90]], columns=["weight"]) ))
            #prediction check for 90kg random weight
            #if instead of dataframe we just passed [90] it'll be a 1D array and as said features cant be 1D
print(f"Predicted height for 90kg weight: {prediction[0]}")


## Performance Metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(np.sqrt(mean_absolute_error(y_test,y_pred)))

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("Rsquared: ",score)
print("AdjustedRsquared: ",1-(1-score)*(len(y_test)-1)/(len(y_test)-1-X_test.shape[1]))

# ## OLS Linear Regression
# import statsmodels.api as sm
# model=sm.OLS(y_train,X_train).fit()
# # prediction=model.predict(X_test)
# # print(prediction)
# # print(model.summary()) ##you can see here the prediction of coeff by ols is same 

# prediction = model.predict((scaler.transform(pd.DataFrame([[90]], columns=["weight"]) )))
#             #prediction check for 90kg random weight
#             #if instead of dataframe we just passed [90] it'll be a 1D array and as said features cant be 1D
# print(f"Predicted height for 90kg weight: {prediction[0]*scaler.scale_+scaler.mean_}")
import statsmodels.api as sm

# Add a constant term to the standardized independent variables
# X_train_with_const = sm.add_constant(X_train)  # Use standardized X_train
# X_test_with_const = sm.add_constant(X_test)    # Use standardized X_test

# Fit the OLS model on standardized data
model = sm.OLS(y_train, X_train).fit()

prediction_scaled = model.predict((scaler.transform( pd.DataFrame([[90]], columns=["weight"]))))
prediction = prediction_scaled * scaler.scale_ + scaler.mean_  # Reverse standardization
print(f"Predicted height for 90kg weight (using OLS, standardized): {prediction[0]}")