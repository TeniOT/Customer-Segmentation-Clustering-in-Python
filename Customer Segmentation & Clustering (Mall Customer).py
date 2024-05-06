#!/usr/bin/env python
# coding: utf-8

# In[93]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[94]:


df = pd.read_csv("C:/Users/toluf/OneDrive/Desktop/DA - Project/Mall customer seg/Mall_Customers.csv")


# In[ ]:





# #### Data View

# In[95]:


df.head()


# In[96]:


df.shape


# In[97]:


df.describe()


# In[98]:


df


# In[99]:


df.isnull()


# In[100]:


df.isnull().sum()


# In[101]:


df.nunique()


# In[102]:


df.dtypes


# In[ ]:





# ### Univariate Analysis

# In[103]:


df.describe()


# In[ ]:





# In[12]:


#sample with 'distplot'


# In[104]:


sns.distplot(df['Annual Income (k$)']);


# In[14]:


#using for loops to view 'int' (number) columns


# In[76]:


#check all columns
df.columns


# In[105]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[ ]:





# In[106]:


#sample with 'kdeplot'
sns.kdeplot(df['Annual Income (k$)'],shade=True);
#sns.kdeplot(x=df['Annual Income (k$)'],shade = True, hue = df['Gender']);
#OR
# melted_df = df.melt(id_vars='Gender', value_vars=['Annual Income (k$)'])
#sns.kdeplot(data=melted_df, x='value', hue='Gender', shade=True)


# In[107]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(data=df,x=i,shade=True,hue='Gender')


# In[ ]:





# In[19]:


#usingboxplot


# In[108]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender',y=df[i])


# In[ ]:





# In[109]:


df['Gender'].value_counts()


# In[110]:


#change to percentage
df['Gender'].value_counts(normalize=True)


# In[ ]:





# ## Bivariate Analysis

# In[111]:


sns.scatterplot(data=df,x='Annual Income (k$)', y='Spending Score (1-100)')


# In[ ]:





# In[24]:


#use pairplot to gather and view all possible bivariate analysis data together rather then pairing for each diff. columns


# In[112]:


sns.pairplot(df)


# In[113]:


#looking at this, might be agood idea to drop the CustomerID column - it's irrelevant data thats adds no value

df=df.drop('CustomerID',axis=1)
sns.pairplot(df)


# In[114]:


sns.pairplot(df,hue='Gender')


# In[ ]:





# In[28]:


df.groupby(['Gender'])[['Age', 'Annual Income (k$)','Spending Score (1-100)']].mean()


# In[ ]:





# In[29]:


##df=df.drop('Gender',axis=1)
df.apply(lambda x: x.factorize()[0]).corr(method='pearson')


# In[ ]:





# In[30]:


correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True, cmap ='coolwarm')

plt.show()


# In[ ]:





# ## Data Analysis

# ### Clustering - Univariate, Bivariate, Multivariate

# In[85]:


##clustering1 = KMeans(n_clusters=6) - started with 6 to find out best number using the elbow method
clustering1 = KMeans(n_clusters=3)


# In[ ]:





# #### Univariate clustering

# In[86]:


clustering1.fit(df[['Annual Income (k$)']])


# In[87]:


clustering1.labels_


# In[ ]:





# In[88]:


#to make sense of this
df['Income Cluster'] = clustering1.labels_
df.head()


# In[89]:


df['Income Cluster'].value_counts()


# In[90]:


clustering1.inertia_


# In[91]:


inertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[38]:


inertia_scores


# In[39]:


#using the elbow method - can see the elbow curves at 3, so change n_clusters to 3

plt.plot(range(1,11),inertia_scores)


# In[ ]:





# In[40]:


#did this to make it easier to copy and paste in code
df.columns


# In[ ]:





# In[41]:


df.groupby('Income Cluster')[['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']].mean()


# In[ ]:





# In[42]:


##to compare relationship between Annual Income and Spending Score


# #### Bivariate Clustering

# In[43]:


#clustering2 = KMeans()  - started with () to find out best number using the elbow method
clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[ ]:





# In[44]:


inertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),inertia_scores2)

#using the elbow method - can see the elbow curves at 5, so change n_clusters to 5


# In[ ]:





# In[45]:


#code to check cluster centres to see where x and y crosses
clustering2.cluster_centers_


# In[ ]:





# In[46]:


#give aliases
centers = pd.DataFrame(clustering2.cluster_centers_)
centers


# In[47]:


#rename
centers.columns = ['x','y']


# In[ ]:





# In[ ]:





# In[48]:


#visualise the bivariate clustering - best to use scatter plot
sns.scatterplot(data=df, x ='Annual Income (k$)', y ='Spending Score (1-100)')


# In[49]:


#plt.figure(figsize=(10,8)) - e.g. to increase or decrease size
#plt.scatter(x=centers['x'], y=centers['y'],s=100 (-size),c='black' (-colour),marker='*') - to highlight the centers
plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)', y ='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
#plt.savefig('clustering_bivariate.png')


# In[ ]:





# In[50]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'])


# In[51]:


#percentage
pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[ ]:





# In[52]:


#checking with Age 
#-can see lowest mean Age (cluster 3) cc high Spending score, need further resaerch - e.g. Sales, spending on games etc.)
#-highest annual income of 88 on cluster 4, mean age 41, but low spending score, need to focus campaign on
df.groupby('Spending and Income Cluster')[['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']].mean()


# In[ ]:





# #### Multivariate Clustering

# In[53]:


from sklearn.preprocessing import StandardScaler


# In[54]:


scale = StandardScaler()


# In[55]:


#for scaling we need all variables to be numbers
#1) use 'replace'
#2) use pandas with 'get_dummies'- we're using this


# In[56]:


df2 = pd.get_dummies(df,drop_first=True,dtype=float)
df2.head()


# In[57]:


df2.columns


# In[58]:


df2 = df2[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]
df2.head()


# In[59]:


df2 = scale.fit_transform(df2)


# In[60]:


df2 = pd.DataFrame(scale.fit_transform(df2))
df2.head()


# In[61]:


inertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(df2)
    inertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),inertia_scores3)


# In[ ]:





# In[62]:


df


# In[63]:


df.to_csv('clustering.csv')


# In[ ]:




