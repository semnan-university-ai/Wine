#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Wine
# dataset link : http://archive.ics.uci.edu/ml/datasets/Wine
# email : amirsh.nll@gmail.com


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:



col_names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', ' Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
wine =pd.read_csv("wine.csv",header=None, names=col_names)


# In[3]:


wine.head()


# In[4]:


inputs =wine.drop('class',axis='columns')
target = wine['class']


# In[5]:


inputs


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3)


# In[15]:



from sklearn.neighbors import KNeighborsClassifier
k=[1,3,5,7,9]
for i in range(len(k)):
 
    knn = KNeighborsClassifier(n_neighbors=k[i])

 
    knn.fit(X_train, y_train)

  
    y_pred = knn.predict(X_test)

    from sklearn import metrics
    print("Accuracy for k = ",k[i]," : ",metrics.accuracy_score(y_test, y_pred))


# In[64]:





# In[ ]:




