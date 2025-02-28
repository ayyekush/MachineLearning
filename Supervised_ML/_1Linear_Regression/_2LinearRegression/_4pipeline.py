import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
X=6*np.random.rand(100,1)-3
y=0.5 * X**2 + 1.5*X +np.random.randn(100,1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=69)

def poly_regression(degree):
    X_new=np.linspace(-3,3,200).reshape(200,1)
    poly_features=PolynomialFeatures(degree=degree,include_bias=True)
    lin_reg=LinearRegression()
    poly_regression=Pipeline([
        ("poly_features",poly_features),
        ("lin_reg",lin_reg)
    ])
    poly_regression.fit(X_train,y_train)
    y_pred_new=poly_regression.predict(X_new)

    #plotting prediction line
    plt.plot(X_new, y_pred_new, color="red", label="degree "+str(degree), linewidth=2)
    plt.scatter(X_train, y_train, color="blue")
    plt.scatter(X_test, y_test, color="red")
    plt.xlabel("X"); plt.ylabel("Y"); plt.legend()
    plt.axis([-4,4,-4,10])
    plt.show()

poly_regression(15)