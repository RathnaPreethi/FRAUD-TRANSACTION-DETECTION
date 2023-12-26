#!/usr/bin/env python
# coding: utf-8

# # FRAUD TRANSACTION DETECTION

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[32]:


credit_card_data = pd.read_csv('creditcard.csv')


# In[33]:


credit_card_data.head()


# In[34]:


credit_card_data.tail()


# In[35]:


credit_card_data.info()


# In[36]:


credit_card_data.isnull().sum()


# In[37]:


from sklearn.impute import SimpleImputer


columns_with_missing = credit_card_data.columns[credit_card_data.isnull().any()]


if not columns_with_missing.empty:
    
    imputer = SimpleImputer(strategy='median')
    
   
    credit_card_data[columns_with_missing] = imputer.fit_transform(credit_card_data[columns_with_missing])
    
   
    print(credit_card_data.isnull().sum())
else:
    print("No missing values found in the DataFrame.")


# In[38]:


credit_card_data['Class'].value_counts()


# In[39]:


legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[40]:


print(legit.shape)
print(fraud.shape)


# In[41]:


legit.Amount.describe()


# In[42]:


fraud.Amount.describe()


# In[43]:


credit_card_data.groupby('Class').mean()


# In[44]:


legit_sample = legit.sample(n=106)


# In[45]:


new_dataset = pd.concat([legit_sample, fraud], axis = 0)


# In[46]:


new_dataset.head()


# In[47]:


new_dataset.tail()


# In[48]:


new_dataset['Class'].value_counts()


# In[49]:


X = new_dataset.drop(columns = 'Class', axis = 1) 
Y = new_dataset['Class'] 


# In[50]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 2)


# In[51]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


# In[52]:


X_train_pred = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_pred, Y_train)


# In[53]:


print("Accuracy on training data: ", training_data_accuracy)


# In[54]:


X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)


# In[55]:


print("Accuracy on training data: ", test_data_accuracy)


# In[57]:


data_to_plot = credit_card_data['Amount'].head(10)
x_labels = credit_card_data.index[:10]
plt.bar(x_labels, data_to_plot)
plt.xlabel('Transaction Index')
plt.ylabel('Amount')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




