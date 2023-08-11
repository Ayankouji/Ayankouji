#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# In[4]:


df = pd.read_csv('C:\countries-aggregated_csv.csv')


# In[6]:


df.head(10)


# In[8]:


df['Date'] = pd.to_datetime(df['Date'])


# In[9]:


df.head()


# In[10]:


df.describe()


# In[11]:


df.info()


# In[13]:


df = df.fillna('NA')


# In[14]:


df.info()


# In[15]:


df.head(10)


# In[18]:


df2 = df.groupby('Country')[['Confirmed','Deaths','Recovered']].sum().reset_index()


# In[23]:


df2 = df.groupby(['Country','Date'])[['Confirmed','Deaths','Recovered']].sum().reset_index()


# In[24]:


df2


# In[25]:


df3 = df2[df2['Confirmed']>100]


# In[26]:


df3


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


x = np.linspace(0,10,1000)
y = np.sin(x)
plt.plot(x,y)


# In[31]:


plt.scatter(x[::10],y[::10],color = "red")


# In[32]:


plt.plot(x,y,color= 'b')
plt.plot(x,np.cos(x),color = 'g')


# In[33]:


countries = df3['Country'].unique()
len(countries)

for idx in range(0,len(countries)):
    C = df3[df3['Country']==countries[idx]].reset_index()
    plt.scatter(np.arange(0,len(C)),C['Confirmed'],color = 'blue', label = 'Confiremed')
    plt.scatter(np.arange(0,len(C)),C['Recovered'],color = 'green', label = 'Recovered')
    plt.scatter(np.arange(0,len(C)),C['Deaths'],color = 'red', label = 'Deaths')
    plt.title(countries[idx])
    plt.xlabel('Days Since the first suspect')
    plt.ylabel('NNumber of Cases')
    plt.legend()
    plt.show()
# In[41]:


df4 = df3.groupby(['Date'])[['Confirmed','Deaths','Recovered']].sum().reset_index()


# In[40]:


for idx in range(0,len(countries)):
    C = df3[df3['Country']==countries[idx]].reset_index()
    plt.scatter(np.arange(0,len(C)),C['Confirmed'],color = 'blue', label = 'Confiremed')
    plt.scatter(np.arange(0,len(C)),C['Recovered'],color = 'green', label = 'Recovered')
    plt.scatter(np.arange(0,len(C)),C['Deaths'],color = 'red', label = 'Deaths')
    plt.title(countries[idx])
    plt.xlabel('Days Since the first suspect')
    plt.ylabel('NNumber of Cases')
    plt.legend()
    plt.show


# 

# In[ ]:




