
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


train_data = pd.read_csv('train_set.csv')
print(train_data.shape)
train_data.head()


# In[4]:


y_1 = train_data[train_data["y"] == 1]
y_0 = train_data[train_data["y"] == 0]
print(len(y_1), len(y_0))


# In[5]:


pdays_neg = train_data[train_data['pdays'] == -1]
print(len(pdays_neg))


# In[6]:


train_job = train_data["job"].value_counts()
train_job


# In[8]:


import seaborn as sns
sns.barplot(y=train_job.index, x=train_job.values)


# In[9]:


train_job_y = train_data.groupby(["job","y"])
train_job_y_counts = train_job_y.size().unstack()
#type(train_job_y_counts)
# train_job_y_counts.sum(1).nlargest(10)
train_job_y_counts = train_job_y_counts.stack()
train_job_y_counts.name = "total"
train_job_y_counts = train_job_y_counts.reset_index()
train_job_y_counts


# In[10]:


def normal_total(group):
    group['norm'] = group.total/group.total.sum()
    return group


# In[11]:


results = train_job_y_counts.groupby('job').apply(normal_total)


# In[12]:


sns.barplot(x='total',y='job',hue='y',data=train_job_y_counts)


# In[13]:


train_marital_y = train_data.groupby(["marital","y"]).size()
train_marital_y.name = 'total'
train_marital_y = train_marital_y.reset_index()
train_marital_y = train_marital_y.groupby('marital').apply(normal_total)
sns.barplot(x='marital',y='norm',hue='y',data=train_marital_y)


# In[14]:


train_education_y = train_data.groupby(["education","y"]).size().unstack()
train_education_y.plot.bar()


# In[15]:


train_poutcome_y = train_data.groupby(["poutcome","y"]).size().unstack()
train_poutcome_y.plot.bar()


# train_contact_y = train_data.groupby(["contact","y"]).size().unstack()
# train_contact_y.plot.bar()
