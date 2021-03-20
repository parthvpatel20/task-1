#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Importinng the data 
import_data = pd.read_csv(r'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print('Data has been imported successfully')

# Displaying number of rows and number of columns present in the data
import_data.shape


# In[4]:


# Let's have a brief overview regarding the data
import_data.head


# In[5]:


# showcasing the statistical details of the data
import_data.describe()


# In[6]:


import_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours Vs Percentage of Students studying')
plt.xlabel('Number of Hours they Study')
plt.ylabel('Percentage of students')
plt.show()


# In[7]:


# Dividing the data in Attributes and Labels
hours = import_data.iloc[:, :-1].values
scores = import_data.iloc[:, 1].values


# In[11]:


from sklearn.model_selection import train_test_split
hours_train, hours_test, scores_train, scores_test = train_test_split(hours, scores,test_size=0.2, random_state=0)


# In[12]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(hours_train,scores_train)


# In[13]:


print(regression.intercept_)


# In[14]:


print(regression.coef_)


# In[15]:


# Let's Plot the regression line
line_plot = regression.coef_ * hours + regression.intercept_ 
plt.scatter(hours,scores)
plt.title('Regression Line Graph')
plt.xlabel('Number of Hours they Study')
plt.ylabel('Percentage of students')
plt.plot(hours,line_plot, c='red')
plt.show()


# In[16]:


# Making Predictions
score_pred = regression.predict(hours_test)

#Comparing Actual vs Predicted Data
df = pd.DataFrame({'Actual Data': scores_test, 'Predicted Data': score_pred})


# In[17]:


df


# In[18]:


plt.scatter(scores_test,score_pred)
plt.title('Actual Vs Predicted')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.show()


# In[19]:


no_of_hours = np.array(9.25).reshape(1,1)
predict_score = regression.predict(no_of_hours)
print(f'Number of hours = {no_of_hours}')
print(f'Score = {predict_score}')


# In[21]:


# At last,We need to check how well this model works
from sklearn import metrics
print('Mean Absolue Error: ', metrics.mean_absolute_error(scores_test,score_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(scores_test,score_pred))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_absolute_error(scores_test,score_pred)))


# In[ ]:
