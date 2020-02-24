#!/usr/bin/env python
# coding: utf-8

# # project 1 ~ Santander Future Transaction Prediction

# In[1]:


#loading libaries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm


# In[2]:


#set working directory
os.chdir("C:/python")


# In[3]:


#check wd
os.getcwd()


# In[4]:


#load the data
train = pd.read_csv("train.csv")


# #####data exploration #####

# In[7]:


#datatypes of data
train.dtypes


# In[6]:


#converting datatype
train['target'] = train['target'].astype(object)

#variables and observations
train.shape


# ############   DATA PRE-PROCSSING   ##############

# In[8]:


###### Missing Value Analysis ######
train.isnull().sum()

#here we can see no missing values there so, no need to apply whole process


# In[9]:


### Outlier Analysis ###
#plot boxplot
get_ipython().run_line_magic('matplotlib', 'inline')

plt.boxplot(train['var_121'])


# In[10]:


#select only numeric
cnames = train.select_dtypes(include=np.number)


# In[11]:


#finding & removing outliers
for i in cnames:
    print(i)
    q75, q25 = np.percentile(train.loc[:,i], [75,25])
    iqr = q75 - q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    #print(min)
    #print(max)
 
    train = train.drop(train[train.loc[:,i] < min].index)
    train = train.drop(train[train.loc[:,i] > max].index)


# In[12]:


train.shape


# In[13]:


###### Feature Selection ######

#select only numeric variabe
cnames = train.select_dtypes(include=np.number)

cnames


# In[14]:


#heatmap

#dimensions of heatmap
f, ax = plt.subplots(figsize=(100, 7))

#correlation matrix
corr = cnames.corr()

#plot
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
           square=True, ax=ax)

#no need to drop any variable as all are important.no co-variance between independent variables


# In[15]:


######### Feature scaling #########

#plot histogram to check normalisation
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(train['var_21'], bins='auto')

#normally distributed


# In[16]:


#select only numeric variable
cnames = train.select_dtypes(include=np.number)


# In[17]:


#standardisation
for i in cnames:
    train[i] = (train[i] - train[i].mean())/train[i].std()


# ####### MODELLING  ########

# In[26]:


train.shape


# In[23]:


#drop a variable
train = train.drop("ID_code", axis=1)


# In[36]:


train = train.astype(float)


# In[38]:


# divide data into train & test
sample = np.random.rand(len(train)) < 0.8

train_df = train[sample]
test = train[~sample]


# ######logistic regression #######

# In[42]:


# select independent variables
train_cols = train.columns[1:201]


# In[43]:


logit = sm.Logit(train_df['target'], train_df[train_cols]).fit()


# In[45]:


logit.summary()


# In[46]:


# prediction
test["Actual_prob"] = logit.predict(test[train_cols])


# In[47]:


#converting predicted probabilities to 0 & 1
test['ActualVal'] = 1
test.loc[test.Actual_prob < 0.5, 'ActualVal'] = 0


# In[48]:


test.head()


# In[49]:


#Confusion matrix logistic regression
CM = pd.crosstab(test['target'], test['ActualVal'])

#save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]


# In[50]:


#logistic regression
CM


# In[52]:


315/(315+3015)


# In[38]:


Accuracy = (TP+TN)/(TP+TN+FP+FN)

FNR = FN/(FN+TP)

Accuracy = 59.98
FNR = 92.94


# #####Naive bayes #####

# In[53]:


#independent & dependent variables
x_train = train.iloc[:,1:201]
y_train = train.iloc[:,0]
x_test = test.iloc[:,1:201]
y_test = test.iloc[:,0]


# In[54]:


NB_model = GaussianNB().fit(x_train, y_train)


# In[55]:


#prediction
NB_predictions = NB_model.predict(x_test)


# In[56]:


#Confusion matrix naive bayes
CM = pd.crosstab(y_test, NB_predictions)


# In[57]:


#save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]


# In[58]:


CM


# In[91]:


#naive bayes
Accuracy = 92.47

FNR = 64.08


# #####Random forest #####

# In[65]:


train_sample = train.sample(85000)


# In[69]:


# divide into train & test
sample = np.random.rand(len(train_sample)) < 0.8

train_df = train_sample[sample]
test = train_sample[~sample]


# In[74]:


#independent & dependent variables
x_train = train_df.iloc[:,1:201]
y_train = train_df.iloc[:,0]
x_test = test.iloc[:,1:201]
y_test = test.iloc[:,0]


# In[75]:


RF_model = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)


# In[76]:


#prediction
RF_predictions = RF_model.predict(x_test)

    


# In[78]:


CM = confusion_matrix(y_test,RF_predictions)


# In[79]:


CM


# In[80]:


#Confusion matrix random forest
CM = pd.crosstab(y_test, RF_predictions)


# In[81]:


CM


# In[ ]:


#randpm forest

accuracy = 90.58
FNR = 1 (can not find as TP is zero & FP is zero)


#here we will freeze naive bayes model as it gives high accuracy and min. FNR which is our concern


# # ~prediction on test.csv

# In[84]:


#importing large test data
santander = pd.read_csv("test.csv")


# In[87]:


#split numeric variables & ID_code
ID_code = santander[santander.columns[0]]
santander = santander.drop("ID_code", axis=1)


# In[90]:


### Missing Value Analysis ###

santander.isnull().sum()

#no missing value found so no need to do whole process


# #####Feature Scaling #######

# In[95]:


#plot histogram to check normalisation
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(santander['var_21'], bins='auto')

#normally distributed


# In[96]:


#select only numeric
cnames = santander.select_dtypes(include=np.number)


# In[99]:


#standardisation
for i in cnames:
    santander[i] = (santander[i] - santander[i].mean())/santander[i].std()


# In[102]:


#prediction
NB_predictions_test = NB_model.predict(santander)


# In[104]:


#converting target float to int
NB_predictions_test = NB_predictions_test.astype(int)


# In[106]:


NB_predictions_test = pd.DataFrame(NB_predictions_test)


# In[108]:


#column bind target results with ID_code
ID_code_target = pd.concat([ID_code, NB_predictions_test], axis=1, ignore_index=True)


# In[109]:


#renaming columns
ID_code_target.columns = ['ID_code','Target']


# In[110]:


#saving output in excel format
ID_code_target.to_excel("Target  value final - Python.xlsx", index = False)


# In[ ]:




