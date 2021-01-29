#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Wine
# dataset link : http://archive.ics.uci.edu/ml/datasets/Wine
# email : amirsh.nll@gmail.com


# In[27]:


import matplotlib.pyplot as plt
import pandas as pd


# In[21]:



col_names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', ' Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
wine =pd.read_csv("wine.csv",header=None, names=col_names)


# In[22]:


wine.head()


# In[28]:



inputs =wine.drop('class',axis='columns')
target = wine['class']


# In[24]:


inputs


# In[29]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3,random_state=109) 


# In[30]:



from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)


# In[31]:


from sklearn import metrics


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




