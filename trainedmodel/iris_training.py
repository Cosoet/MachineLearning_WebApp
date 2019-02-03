#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pickle
import requests, json


# In[9]:


iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[11]:


rfc = RandomForestClassifier(n_estimators=100)


# In[12]:


rfc.fit(X_train, y_train)


# In[14]:


print(accuracy_score(y_test, rfc.predict(X_test)))
print(classification_report(y_test, rfc.predict(X_test)))


# In[15]:


pickle.dump(rfc, open("iris_rfc.pkl", "wb"))


# In[18]:


#my_random_forest = pickle.load(open("iris_rfc.pkl", "rb"))


# In[ ]:




