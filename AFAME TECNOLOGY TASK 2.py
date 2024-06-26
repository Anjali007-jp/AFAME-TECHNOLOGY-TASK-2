#!/usr/bin/env python
# coding: utf-8

# ## AFAME TECHNOLOGY
# ## TITANIC 

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data

# In[21]:


data=pd.read_csv("Titanic_Dataset.csv")


# In[22]:


data


# In[23]:


data.head()


# # EDA

# ## Missing Value

# In[24]:


data.isnull()


# In[25]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[26]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=data)


# In[27]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=data,palette='RdBu_r')


# In[28]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=data,palette='rainbow')


# In[29]:


sns.distplot(data['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[30]:


data['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[31]:


sns.countplot(x='SibSp',data=data)


# In[32]:


data['Fare'].hist(color='green',bins=40,figsize=(8,4))


# # Cufflinks for plots

# In[33]:


pip install cufflinks


# In[34]:


import cufflinks as cf
cf.go_offline()


# In[19]:


data['Fare'].iplot(kind='hist',bins=30,color='green')


# In[35]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=data,palette='winter')


# In[36]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[37]:


data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)


# In[38]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[39]:


data.drop('Cabin',axis=1,inplace=True)


# In[40]:


data.head()


# In[41]:


data.dropna(inplace=True)


# # Converting Categorical Features

# In[42]:


data.info()


# In[43]:


pd.get_dummies(data['Embarked'],drop_first=True).head()


# In[45]:


sex = pd.get_dummies(data['Sex'],drop_first=True)
embark = pd.get_dummies(data['Embarked'],drop_first=True)


# In[46]:


data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[47]:


data.head()


# In[49]:


data=pd.concat([data,sex,embark],axis=1)


# In[50]:


data.head()


# # Building a Logistic Regression model

# ### Train Test Spilt

# In[52]:


data.drop('Survived',axis=1).head()


# In[53]:


data['Survived'].head()


# In[54]:


from sklearn.model_selection import train_test_split


# In[56]:


x_train,x_test,y_train,y_test = train_test_split(data.drop('Survived',axis=1),data['Survived'],test_size=0.30,random_state=101)


# # Training and Predicting

# In[59]:


from sklearn.linear_model import LogisticRegression


# In[61]:


logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[62]:


predictions = logmodel.predict(x_test)


# In[63]:


from sklearn.metrics import confusion_matrix


# In[64]:


accuracy=confusion_matrix(y_test,predictions)


# In[65]:


accuracy


# In[66]:


from sklearn.metrics import accuracy_score


# In[67]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[68]:


predictions


# In[ ]:




