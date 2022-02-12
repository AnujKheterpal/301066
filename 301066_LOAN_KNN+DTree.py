#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
pd.options.display.max_columns =None
pd.options.display.max_rows =None


# In[2]:


df  = pd.read_csv("C://Users//user//OneDrive//Desktop//bank_cleaned.csv")
df.head()
df.tail()


# In[3]:


## DATA PRE-PROCESSING

#Finding out missing values in each column
print(df.isna().sum())
missing_status=df.isna().sum()
##print(df.isnull().sum())


# In[4]:


df[df.duplicated(keep = 'last')] 


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.nunique()


# In[8]:


df.columns


# In[9]:


df.describe()


# In[11]:


df.groupby(['education']).size()


# In[12]:


df.sample(n=10)


# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.job = le.fit_transform(df.job)
df.marital = le.fit_transform(df.marital)
df.loan = le.fit_transform(df.loan)

df.head(5)


# In[15]:


df = df.fillna(0)


# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

x= df[['job','marital']]
y = df[['loan']]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
y_train.head()


# In[40]:


st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  
y_test


# In[41]:


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train, y_train)


# In[42]:


y_pred = knn.predict(x_test)


# In[43]:


confusion_matrix(y_test, y_pred)


# In[44]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[45]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred)


# In[47]:


from matplotlib import pyplot as plt
error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(x_train,y_train)
 pred_i = knn.predict(x_test)
 #print (pred_i)
 #print (1-accuracy_score(y_test, pred_i))
 error_rate.append(1-accuracy_score(y_test, pred_i))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate))+1)


# In[48]:


knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
knn.fit(x_train, y_train)


# In[49]:


y_pred = knn.predict(x_test)
accuracy_score(y_test, y_pred)


# In[50]:


import os
from sklearn.compose import ColumnTransformer as ct
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.tree import DecisionTreeClassifier as dt
###################### Random Forest ############################
from sklearn.ensemble import RandomForestClassifier 


# In[51]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[52]:


sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[53]:


classifier = dt(criterion = 'entropy', random_state = 0)
#classifier = dt(criterion = 'entropy', random_state = 0, max_depth=4)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)


# In[54]:


print("Predicted Values : ",Y_pred[1:50])


# In[55]:


print("Accuracy:",accuracy_score(Y_test, Y_pred))


# In[56]:


from sklearn import tree
plt.figure(figsize=(12,7))
tree.plot_tree(classifier, filled=True, fontsize=12)


# In[57]:


X_dataframe = x

grr = pd.plotting.scatter_matrix(X_dataframe, figsize=(25, 25), marker='o', hist_kwds={'bins': 20}, s=10, alpha=.8)


# In[58]:


import seaborn as sns
plt.figure(figsize=(25,25))
sns.heatmap(x.corr(), center=0, annot=True)
plt.title("Correlation Map")
plt.show()


# In[59]:


print("Accuracy:",accuracy_score(Y_test, Y_pred))


# In[60]:


#both yield similar accuracy score


# In[ ]:




