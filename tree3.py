#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Wine
# dataset link : http://archive.ics.uci.edu/ml/datasets/Wine
# email : amirsh.nll@gmail.com


# In[8]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[9]:


col_names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', ' Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
wine =pd.read_csv("wine.csv",header=None, names=col_names)


# In[10]:


wine.head()


# In[11]:


inputs =wine.drop('class',axis='columns')
target =wine['class']
target


# In[13]:


input_train,input_test,target_train,target_test=train_test_split(inputs,target,test_size=0.3,random_state=1)


# In[14]:


clf =DecisionTreeClassifier()
clf =clf.fit(input_train,target_train)
y_pred = clf.predict(input_test)


# In[15]:


print ("Accuracy:",metrics.accuracy_score(target_test,y_pred))


# In[ ]:




