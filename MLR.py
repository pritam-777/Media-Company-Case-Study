#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("mediacompany.csv")


# In[3]:


df.head()


# # Checking Duplicates

# In[4]:


sum(df.duplicated(subset='Date'))==0


# In[5]:


df=df.drop('Unnamed: 7',axis=1)


# In[6]:


df.head()


# # Data Inspection

# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


df['Date'] = pd.to_datetime(df['Date'], dayfirst = False )


# In[12]:


df.head()


# In[13]:


df['Day_Of_Week']=df['Date'].dt.dayofweek


# In[14]:


df.head()


# # Exploratory Data Analysis

# In[15]:


sns.boxplot(df['Views_show'])


# In[16]:


df.plot.line(x='Date',y='Views_show')


# In[17]:


sns.barplot(data=df,x="Day_Of_Week",y="Views_show")


# # we can see that Views are more on 'Sunday' and 'Saturday'(weekends) and decline on subsequent days

# In[18]:


ax = df.plot(x="Date", y="Views_show", legend=False)
ax2 = ax.twinx()
df.plot(x="Date", y="Ad_impression", ax=ax2, legend=False, color="r")
ax.figure.legend()


# In[19]:


sns.scatterplot(data=df,x="Ad_impression",y="Views_show")


# # we can see that the views as well as ad impressions show a weekly pattern.

# In[20]:


sns.scatterplot(data=df,x='Views_platform', y = 'Views_show')


# Show views are some what proportionately related to Platform views

# In[21]:


sns.barplot(data=df,x='Cricket_match_india', y='Views_show')


#  Show views slightly declines when there is a cricket match.

# In[22]:


sns.barplot(data = df,x='Character_A', y='Views_show')


#  Presence of Character A improves the show viewership.

# # Model building

# In[23]:


from sklearn.preprocessing import MinMaxScaler


# In[24]:


scaler=MinMaxScaler()


# In[25]:


num_vars = ['Views_show','Visitors','Views_platform','Ad_impression']

df[num_vars] = scaler.fit_transform(df[num_vars])


# In[26]:


df.head()


# In[27]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)


# In[28]:


di = {5:1, 6:1, 0:0, 1:0, 2:0, 3:0, 4:0}
df['weekend'] = df['Day_Of_Week'].map(di)


# In[29]:


df.head()


# In[30]:


X=df[["Visitors","weekend"]]
y=df["Views_show"]


# In[31]:


from sklearn.linear_model import LinearRegression


# In[32]:


lm=LinearRegression()


# In[33]:


lm.fit(X,y)


# In[34]:


import statsmodels.api as sm


# In[35]:


X=sm.add_constant(X)
lm_1=sm.OLS(y,X).fit()
print(lm_1.summary())


# Visitors as well as weekend column are significant.

# In[36]:


X=df[["Visitors","weekend","Character_A"]]
y=df["Views_show"]


# In[37]:


X=sm.add_constant(X)
lm_2=sm.OLS(y,X).fit()
print(lm_2.summary())


# In[38]:


df['Lag_Views'] = np.roll(df['Views_show'], 1)
df.head()


# In[39]:


df.Lag_Views[0]=0


# In[40]:


df.head()


# In[41]:


X=df[["Visitors","weekend","Character_A","Lag_Views"]]
y=df["Views_show"]


# In[42]:


X=sm.add_constant(X)
lm_3=sm.OLS(y,X).fit()
print(lm_3.summary())


# visitor insignificant.

# In[43]:


X = df[['weekend','Character_A','Views_platform']]
y=df["Views_show"]


# In[44]:


X=sm.add_constant(X)
lm_4=sm.OLS(y,X).fit()
print(lm_4.summary())


# In[45]:


X = df[['weekend','Character_A',"Visitors"]]
y=df["Views_show"]


# In[46]:


X=sm.add_constant(X)
lm_5=sm.OLS(y,X).fit()
print(lm_5.summary())


# In[47]:


X = df[['weekend','Character_A',"Visitors","Ad_impression"]]
y=df["Views_show"]


# In[48]:


X=sm.add_constant(X)
lm_6=sm.OLS(y,X).fit()
print(lm_6.summary())


# In[49]:


X = df[['weekend','Character_A',"Ad_impression"]]
y=df["Views_show"]


# In[51]:


X=sm.add_constant(X)
lm_7=sm.OLS(y,X).fit()
print(lm_7.summary())


# In[52]:


df['ad_impression_million'] = df['Ad_impression']/1000000


# In[53]:


X =df[['weekend','Character_A','ad_impression_million','Cricket_match_india']]
y=df["Views_show"]


# In[54]:


X=sm.add_constant(X)
lm_8=sm.OLS(y,X).fit()
print(lm_8.summary())


# In[55]:


X = df[['weekend','Character_A','ad_impression_million']]
y=df["Views_show"]


# In[57]:


X=sm.add_constant(X)
lm_9=sm.OLS(y,X).fit()
print(lm_9.summary())


# # Making predictions using the model

# In[58]:


X = df[['weekend','Character_A','ad_impression_million']]
X = sm.add_constant(X)
Predicted_views = lm_9.predict(X)


# In[59]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(df.Views_show, Predicted_views)
r_squared = r2_score(df.Views_show, Predicted_views)


# In[60]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# # Actual vs Predicted

# In[66]:


c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,df.Views_show, color="#CB4335", linewidth=2.5, linestyle="-")
plt.plot(c,Predicted_views, color="#2ECC71",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Views', fontsize=16)     


# # Error Plot

# In[67]:


c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,df.Views_show-Predicted_views, color="#5DADE2", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=22)               
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Views_show-Predicted_views', fontsize=16) 


# Making predictions using lm_5

# In[68]:


X = df[['weekend','Character_A',"Visitors"]]
X =sm.add_constant(X)
Predicted_views=lm_5.predict(X)


# In[70]:


mse = mean_squared_error(df.Views_show, Predicted_views)
r_squared = r2_score(df.Views_show, Predicted_views)


# In[71]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[73]:


c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,df.Views_show, color="#CA6F1E", linewidth=2.5, linestyle="-")
plt.plot(c,Predicted_views, color="#2ECC71",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Views', fontsize=16)    


# # Error plot

# In[74]:


c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,df.Views_show-Predicted_views, color="#8E44AD", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=22)               
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Views_show-Predicted_views', fontsize=16) 


# Ad Impressions and Character A as the driver variables that could explain the viewership pattern. Based on industry experience, ad impressions are directly proportional to the marketing budget. Thus, by increasing the marketing budget, a better viewership could be achieved. Similarly, Character A’s absence and presence created a significant change in show viewership. Character A’s presence brings viewers to the show. Thus, these two variables could be acted upon to improve show viewership.

# In[ ]:




