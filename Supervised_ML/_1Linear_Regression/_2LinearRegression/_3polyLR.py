import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# a random quadratic eqn and its plotting
X=(np.random.rand(100,1)*6)-3
y=0.5 * X**2 + 1.5*X +np.random.randn(100,1)
# plt.scatter(X,y,color='green')
# plt.xlabel('X dataset')
# plt.ylabel('y dataset')
# plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=69)

### Lets impl a simpleLR
from sklearn.linear_model import LinearRegression
regression_1=LinearRegression()
regression_1.fit(X_train,y_train)

from sklearn.metrics import r2_score
score= r2_score(y_test,regression_1.predict(X_test))
print(score) #so low because a straight line doesnt fit a quad eqn

# plt.plot(X_train,regression_1.predict(X_train),color="red")
# plt.scatter(X_train,y_train)
# plt.show()

#Lets apply polynomial transfromation (basically has x wise value ka sq kar rahe)
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2,include_bias=True)
X_train_poly=poly.fit_transform(X_train)#chek diff btw fittrans and trans
X_test_poly=poly.transform(X_test)

from sklearn.metrics import r2_score
regression_2=LinearRegression()
regression_2.fit(X_train_poly,y_train)
score=r2_score(y_test,regression_2.predict(X_test_poly))
print(score) #much much better score, since we also used x^2
print(regression_2.coef_)

# plt.scatter(X_train,regression_2.predict(X_train_poly))
# plt.scatter(X_train,y_train)
# plt.show() #a much better graph

###Lets apply polynomial transform and add a cube
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3,include_bias=True)
X_train_poly3=poly.fit_transform(X_train)#chek diff btw fittrans and trans
X_test_poly3=poly.transform(X_test)

from sklearn.metrics import r2_score
regression_3=LinearRegression()
regression_3.fit(X_train_poly3,y_train)
score=r2_score(y_test,regression_3.predict(X_test_poly3))
print(score) #almost the same as x^2, means we do not need x^3
print(regression_2.coef_)

# plt.scatter(X_train,regression_3.predict(X_train_poly3))
# plt.scatter(X_train,y_train)
# plt.show() #a much better graph

### Prediction of a new dataset
X_new=np.linspace(-3,3,200).reshape(200,1)#a random linear space with 200 values
X_new_poly=poly.transform(X_new) #poly curtny have deg 3 (last used)
# print(X_new_poly) # 1 is the bias

y_new=regression_3.predict(X_new_poly)
plt.plot(X_new,y_new,color="r",linewidth="2",label="Predictions")
plt.scatter(X_train,y_train,color="blue",label="Training Points")
plt.scatter(X_test ,y_test,color="green",label="Testing Points")
plt.xlabel("X"); plt.ylabel("Y"); plt.legend()
plt.show()