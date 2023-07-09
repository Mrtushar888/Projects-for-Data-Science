#!/usr/bin/env python
# coding: utf-8

# In[55]:


pip install pickle


# In[36]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[37]:


data=pd.read_csv('placement.csv')
data.head(2)


# In[38]:


new=data.iloc[:,1:]
new


# In[39]:


new.isnull().sum()


# In[40]:


new.info()


# In[41]:


new.duplicated().value_counts()


# In[42]:


plt.scatter(new['cgpa'],new['iq'],c=new['placement'])


# In[43]:


x=new.iloc[:,:2]
y=new.iloc[:,-1]


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# In[45]:


scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)


# In[46]:


x_test=scaler.fit_transform(x_test)


# In[47]:


model=LogisticRegression()


# In[48]:


model.fit(x_train,y_train)


# In[49]:


y_pred=model.predict(x_test)


# In[50]:


y_test


# In[51]:


from sklearn.metrics import accuracy_score


# In[52]:


accuracy_score(y_pred,y_test)


# In[53]:


from mlxtend.plotting import plot_decision_regions


# In[54]:


plot_decision_regions(x_train,y_train.values, clf=model,legend=2)


# In[ ]:




