### EDA= Exploratory Data Analysis (diags and shit)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("_0forestDATAcleaned.csv")

#dropping some not neccessary features
df_copy=df.drop(["day","month","year"],axis=1)

#ENCODING of the categories in the classes column (binary encoding)
        #not fire ki jagha 0 aur fire ki jagha 1
# print(df_copy["Classes"].value_counts())
df_copy["Classes"]=np.where(df_copy["Classes"].str.strip()=="not fire",0,1)
                  #np.where(if,do this,else this)
# print(df_copy["Classes"].value_counts())
df_copy.to_csv("_0forestDATAcleaned.csv",index=False)

#Density plot for all features
# df_copy.hist(bins=50,figsize=(20,15))
# plt.show()

#Pie chart
#whatever check it out

### CORRELATION
print(df_copy.corr())
# sns.heatmap(df_copy.corr()) #this shit cool as hell
# plt.show()

# Box plot
    #to showcase the outliers
sns.boxplot(df["FWI"])
plt.show()
