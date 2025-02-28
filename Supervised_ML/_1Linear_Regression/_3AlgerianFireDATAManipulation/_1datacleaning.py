import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset= pd.read_csv("_0forestDATA.csv",header=1)#header1 removed header
#contains data for bahai regrion and side-ebel region.

#to distinguish between regions, adding a new column Region
dataset.loc[:122,"Region"]=0
dataset.loc[122:,"Region"]=1
# print(dataset.info()) #tells us that Region in implicitly float, but float takes space 
                            #so we convert it into int
dataset[["Region"]]=dataset[["Region"]].astype(int)

### Remove the null values
# print(dataset[dataset.isnull().any(axis=1)]) #print rows where any val is null
dataset=dataset[~dataset.isnull().any(axis=1)] #to remove null rows (tilde is bitwise not)
                                                    #or we can just dataset.dropna() to drop null val rows :)
# dnow the side-ebel region line is removed, now just beneath it a repeated legend was there so remove it too:
dataset=dataset.drop(index=123)
dataset = dataset.reset_index(drop=True) #since indexes are preserved, any removing of row creates 
                                                #a gap, so we reset indices

## fix spaces in column names cuz why not
dataset.columns=dataset.columns.str.strip()

##change suitable columns from object type to integer type too(like month, day)
dataset[["month","day","year","Temperature","RH","Ws"]]=dataset[["month","day","year","Temperature","RH","Ws"]].astype(int)
# ##changing everything else to float (except classes)
for j in [i for i in dataset.columns if dataset[i].dtype=="O"]: #o means object
    if j!="FFMC" and j!="Classes":
        dataset[j]=dataset[j].astype(float)


## lets save the cleaned dataset
dataset.to_csv("_0forestDATAcleaned.csv",index=False) #indfalse to NOT save index