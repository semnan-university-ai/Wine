#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Wine
# dataset link : http://archive.ics.uci.edu/ml/datasets/Wine
# email : amirsh.nll@gmail.com


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split


# In[35]:



col_names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', ' Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
wine =pd.read_csv("wine.csv",header=None, names=col_names)


# In[36]:


wine.head()


# In[32]:



inputs =wine.drop('class',axis='columns')
target = wine['class']


# In[38]:


inputs


# In[39]:


input_train,input_test,target_train,target_test=train_test_split(inputs,target,test_size=0.3,random_state=1)


# In[40]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(6,7), max_iter=1000)
mlp.fit(input_train, target_train)


# In[31]:


from sklearn.metrics import accuracy_score
predictions_train = mlp.predict(input_train)

print("accuracy for train data: ", accuracy_score(predictions_train, target_train))
predictions_test = mlp.predict(input_test)
print(target_test)
print(predictions_test)
print("accuracy for test data: ", accuracy_score(predictions_test, target_test))


# In[ ]:




