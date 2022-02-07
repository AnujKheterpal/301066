#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


df  = pd.read_csv("C://Users//user//OneDrive//Desktop//bank_cleaned.csv")


# In[13]:


df


# In[6]:


df.columns


# In[17]:


df  = pd.read_csv("C://Users//user//OneDrive//Desktop//bank_cleaned.csv")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.job = le.fit_transform(df.job)
df.marital = le.fit_transform(df.marital)
df.loan = le.fit_transform(df.loan)

print(df)

    


# In[19]:


X= df[['job','marital']]


# In[20]:


Y = df[['loan']]


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
y_train.head()


# In[22]:


X_train.head()


# In[23]:


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)


# In[24]:


y_pred = knn.predict(X_test)


# In[25]:


confusion_matrix(y_test, y_pred)


# In[26]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[27]:


from sklearn.metrics import precision_recall_fscore_support


# In[28]:


precision_recall_fscore_support(y_test, y_pred)


# In[29]:


from sklearn.metrics import precision_score


# In[30]:


precision_score(y_test, y_pred)


# In[31]:


from sklearn.metrics import recall_score


# In[32]:


recall_score(y_test, y_pred)


# In[33]:


from sklearn.metrics import f1_score


# In[34]:


f1_score(y_test, y_pred)


# In[35]:


error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
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


# In[36]:


knn = KNeighborsClassifier(n_neighbors=8, metric='euclidean')
knn.fit(X_train, y_train)


# In[37]:


y_pred = knn.predict(X_test)


# In[38]:


accuracy_score(y_test, y_pred)


# In[ ]:




