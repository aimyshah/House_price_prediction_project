#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error 
# Importing various models for testing which one works best with the given dataset using cross validation.
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor


# In[2]:


house_prices_dataset = pd.read_csv('Data(House_price_prediction_project_#3)\Boston_House_Price_dataset.csv')


# In[3]:


print(house_prices_dataset)


# In[4]:


house_prices_dataset.head()


# In[5]:


house_prices_dataset.shape


# In[6]:


# Checking for missing values.
house_prices_dataset.isnull().sum()


# In[7]:


# Statistical measures of the dataset.
house_prices_dataset.describe()


# In[8]:


# Understanding the correlation between various features in the dataset.
correlation = house_prices_dataset.corr()


# In[9]:


# Making a heatmap to understand the correlation.
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# #### Splitting the data and target.

# In[10]:


X = house_prices_dataset.drop(['PRICE'], axis=1)
Y = house_prices_dataset['PRICE']


# In[11]:


print(X)
print(Y)


# ### Testing which model is best suited for this dataset.

# In[12]:


# Defining the list of regression models.
models = [LinearRegression(), DecisionTreeRegressor(), Lasso(), XGBRegressor()]


# In[13]:


# Function to evaluate models using MSE(Mean squared error).
def evaluate_models(X, Y):
    for model in models:
        # Cross validation with MSE as the scoring metric.
        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        cv_scores = cross_val_score(model, X, Y, scoring=mse_scorer)
        # Negate because sklearn scorers return negative values for losses.
        mean_mse = -cv_scores.mean()
        mean_mse = round(mean_mse, 2)
        print('Mean squared error of', model, 'is: ', mean_mse)
        print('---------------------------------------------------------')


# In[14]:


evaluate_models(X, Y)


# ### Inference

# In[15]:


# In this case, the XGBRegressor is performing the best.


# ### Model Training

# #### XGBoost Regressor

# In[16]:


# Splitting the data into training and test data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[17]:


# Loading the model.
model = XGBRegressor()


# In[18]:


# Training the model.
model.fit(X_train, Y_train)


# ### Evaluating the model.

# In[19]:


# Checking predictions on training data.
X_train_predictions = model.predict(X_train)


# In[20]:


# R squared error.
score_1 = metrics.r2_score(Y_train, X_train_predictions)

# Mean Absolute error.
score_2 = metrics.mean_absolute_error(Y_train, X_train_predictions)

print('R squared error: ', score_1)
print('Mean Absolute error: ', score_2)


# ### Visualizing the actual prices and the predicted prices.

# In[21]:


plt.scatter(Y_train, X_train_predictions)
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Actual Price vs Predicted price')
plt.show()


# In[22]:


# Checking predictions on test data.
X_test_predictions = model.predict(X_test)


# In[23]:


# R squared error.
score_1 = metrics.r2_score(Y_test, X_test_predictions)

# Mean Absolute error.
score_2 = metrics.mean_absolute_error(Y_test, X_test_predictions)

print('R squared error: ', score_1)
print('Mean Absolute error: ', score_2)


# ### Making a predictive system

# In[29]:


# Function to get user input for all features.
def get_input_data():
    print('Enter values for the following feature names: ')
    
    feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

    input_data = []
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        input_data.append(value)
        
    return np.array(input_data).reshape(1, -1)
    
input_data = get_input_data()

prediction = model.predict(input_data)

print(f"The predicted house price is: {prediction[0]}")


# In[ ]:




