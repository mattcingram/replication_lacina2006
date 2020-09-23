#!/usr/bin/env python
# coding: utf-8

# # Replication with Python from Jupyter
# ### Matt Ingram
# ### University at Albany, SUNY
# # Replication 1: Lacina (2006)
# ### __Computing Tools__: Python and markdown in JupyterHub
# ### __Subject fields__: political science, international relations, civil war, political violence, conflict studies, peace science
# ### __Methods topics__: data management, descriptive statistics, histograms, OLS

# # Introduction

# This notebook documents a replication of Lacina (2006) in Python from within the Jypyter platform. 
# 
# Python runs natively in Jupyter, so no kernel needs to be installed as was done with Stata. 
# 
# I am using JupyterHub and JupyterLab. JupyterHub is a server-based version of Jupyter that allows central installation of software and multiple users. In academic settings, the advantages are:
# - shared resources to avoid duplication and enhance collaboration
# - secure sign-on with instiutional IDs and passwords
# - remote computing
# - extensions to use JupyterHub as a learning management system, including assigmment management
# 
# JupyterLab is an interface that mimics features of an integrated development environment (IDE), allowing multiple notebooks to be opened at once, side by side, while at the same time being able to view directory, pull-down menus, etc.
# 
# For more information on Jupyter, see: http://jupyter.org/
# 
# For more information on JupyterHub, see: https://github.com/jupyterhub/jupyterhub
# 
# For more information on JupyterLab, see: https://github.com/jupyterlab/jupyterlab

# # Set Environment

# ## Import Python modules (packages)

# In[1]:


import os      # to manage operating system
import pandas as pd  # package to manage data
import io      # manage input/output requests
import requests   # manage web requests
import numpy as np   # core methods package for math and to manage various data objects; many other uses
import matplotlib.pyplot as plt  # plotting library
from sklearn import linear_model   # data science library
import statsmodels.formula.api as sm  # statistics library
import seaborn as sns   # data visualization library
import inspect # contains getsource() function to inspect source code
import platform # to check identifying info of python, e.g., version


# # Set Working Directory

# In[2]:


print(os.getcwd()) # check current working dir
path = '/home1/s/m/mi122167/OpenStats/replication1lacina'
os.chdir(path)
print(os.getcwd()) # ensure cwd changed to desired dir


# # Create sub-directories

# In[3]:


os.makedirs('./code', exist_ok=True)
os.makedirs('./data/original', exist_ok=True)
os.makedirs('./data/working', exist_ok=True)
os.makedirs('./figures', exist_ok=True)
os.makedirs('./tables', exist_ok=True)


# In[4]:


import platform


# In[5]:


platform.python_version()


# # Load Data

# In[6]:


url="http://mattingram.net/teaching/workshops/introR/materials/Lacina_JCR_2006_replication.csv"
s=requests.get(url).content
data=pd.read_csv(io.StringIO(s.decode('utf-8')))
data.iloc[0:5, 0:10]     # .iloc allows indexing within an object, here a dataframe


# In[ ]:


# examine source code of pd.read_csv()
#print(inspect.getsource(pd.read_csv))


# In[7]:


data.shape[0]


# Notice that the observation identifier begins with 0 in Python. This is different from R and Stata.
# 
# In Python, if you want the first element of any object, that element is '0'.
# In R or Stata, the first element is '1'.

# # Summary Statistics

# In[8]:


data = data.loc[:,['battledeadbest', 'lnbdb', 'lnduration', 'lnpop', 'lnmilqual', 'lngdp', 
                   'cw', 'lnmountain', 'democ', 'ethnicpolar', 'relpolar', 'region']]
data.describe()


# # Histograms of outcome of interest

# In[9]:


data['battledeadbest'].hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.show()


# In[10]:


data['lnbdb'].hist(grid=True, bins=int(data.shape[0]/10), rwidth=0.9,
                   color='#607c8e')
plt.show()


# # Histogram with seaborn module

# In[11]:


sns.distplot(data.battledeadbest, hist=True, kde=True, 
             bins=int(data.shape[0]/10), color = 'darkblue', 
             hist_kws={'edgecolor':'black'}, 
             kde_kws={'linewidth': 4})


# In[12]:


sns.distplot(data.lnbdb, hist=True, kde=True, 
             bins=int(data.shape[0]/10), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


# # Linear Regression

# ## OLS with Statsmodels package

# This is Table 2 from original paper.
# 
# Model takes the form of $y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \epsilon$.
# 
# In matrix notation, $y = X\beta + \epsilon$, where y is a vector containing the outcome of interest (lnbdb), $X$ is a matrix of predictors (model matrix), and $\epsilon$ is the error term.

# In[13]:


m1 = sm.ols(formula="lnbdb ~ lnduration + lnpop + lnmilqual + lngdp + cw + lnmountain + democ + ethnicpolar + relpolar", 
                data=data).fit()
m2 = sm.ols(formula="lnbdb ~ lnduration + cw + democ + ethnicpolar", 
                data=data).fit()
m1.summary()


# In[14]:


m2.summary()


# # OLS with Sklearn package

# In[15]:


m1 = linear_model.LinearRegression()
X = data.loc[:,['lnduration', 'lnpop', 'lnmilqual', 'lngdp', 'cw', 'lnmountain', 'democ', 'ethnicpolar', 'relpolar']]
X.head()
y = data.loc[:,['lnbdb']]
y.head()
# same as data.lnbdb[:5]


# In[16]:


np.any(np.isnan(X))
# returns true
data = data.dropna()
X = data.loc[:,['lnduration', 'lnpop', 'lnmilqual', 'lngdp', 'cw', 'lnmountain', 'democ', 'ethnicpolar', 'relpolar']]
y = data.loc[:,['lnbdb']]


# In[17]:


m1 = linear_model.LinearRegression()
m1.fit(X, y)


# In[18]:


print(m1.intercept_)
print(m1.coef_)


# In[19]:


pd.DataFrame(list(zip(X.columns, np.transpose(m1.coef_))), columns = ['predictors', 'coefficients'])


# In[20]:


plt.scatter(data.lnduration, data.lnbdb, color='blue')
yhat = m1.predict(X)
plt.scatter(data.lnduration, yhat, color='red')
plt.show()


# In[21]:


sns.lmplot(x="lnduration", y="lnbdb", hue="democ", data=data);


# In[22]:


sns.lmplot(x="lnduration", y="lnbdb", hue="region", data=data);


# In[23]:


grid = sns.FacetGrid(data, col="region", hue="region", palette="tab20c",
                     col_wrap=2, height=2)
grid.map(plt.scatter, "lnduration", "lnbdb", marker="o")


# In[ ]:





# In[ ]:




