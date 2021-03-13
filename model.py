#!/usr/bin/env python
# coding: utf-8

# In[40]:


from sklearn.linear_model import LinearRegression
import pandas_datareader as web
from sklearn import preprocessing
import pickle


# In[2]:


df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')


# In[26]:


df['Predictions'] = df['Close'].shift(-1)
df = df.dropna()


# In[27]:


X_train = df[[col for col in df.columns if col != 'Predictions']]


# In[28]:


y_train = df[['Predictions']]


# In[29]:


scaler = preprocessing.StandardScaler()


# In[30]:


X_scaled = scaler.fit_transform(X_train)


# In[31]:


target_scaler = preprocessing.StandardScaler()


# In[32]:


y_scaled = target_scaler.fit_transform(y_train)


# In[36]:


reg = LinearRegression().fit(X_scaled, y_scaled)


# In[39]:


preprocessor = {'standardScaler': [scaler, target_scaler]}
model = {'model': reg}


# In[42]:


with open('preprocessor.pkl',  'wb') as f:
    pickle.dump(preprocessor, f)


# In[43]:


with open('model.pkl',  'wb') as f:
    pickle.dump(model, f)

