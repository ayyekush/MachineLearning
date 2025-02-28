import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("_0forestDATAcleaned.csv")

## Independent And Dependent Feature
X=df.drop("FWI",axis=1) #everything except fwi
y=df["FWI"]

### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

### Feature Selection based on correlation
# sns.heatmap(X_train.corr())
# plt.show()

#check for multicolinearity
def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
correlation(X_train,0.85)#gives the columns names that have correlatiion grater than 85 percent(thrshold val)
X_train.drop(correlation(X_train,0.85),axis=1,inplace=True)
X_test.drop(correlation(X_train,0.85),axis=1,inplace=True)


## Feature Scaling and Standardization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

#Box plots to understand effects of standard scaler
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title("X_train Before Scaling")
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title("X_train After scaling")