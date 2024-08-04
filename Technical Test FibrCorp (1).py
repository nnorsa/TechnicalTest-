#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# Data preparation is the process of preparing data before modeling or analysis. This stage includes importing libraries, reading the dataset, displaying basic dataset information, cleaning the data by removing unnecessary columns, checking for missing values, and deleting duplicate data. These steps ensure that the data is in the proper format and ready for use.

# In[653]:


#import library that required
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[654]:


#change the value display format in the variable 'price_IDR'
pd.options.display.float_format = '{:,.2f}'.format


# In[655]:


#read dataset with excel format
df = pd.read_excel('laptop_price.xlsx')
df


# In[656]:


#display the dataset info
df.info()


# In[657]:


#drop columns that no needed
df = df.drop(columns=['Unnamed: 0'])


# In[658]:


#check missing values of dataset
df.isna().sum()


# In[659]:


#drop duplicates data
df.drop_duplicates(inplace=True)


# Displaying information about the data after removing duplicates. Initially, the dataset contained 1303 entries, and after the duplicate removal process, it now has 1274 entries.

# In[660]:


df.info()


# # Preprocessing

# During the preprocessing stage, I converted certain variables (RAM and Memory) from string data types to numeric and performed encoding using One-Hot Encoding. This method was chosen because it is safer and widely used. One-Hot Encoding creates separate variables for each category, which helps avoid issues related to ordering and makes the data easier for the model to interpret.

# In[661]:


#convert RAM variable from string to numeric
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)


# In[662]:


#display data from RAM variables that have been converted to numeric
df['Ram'].unique()


# In[663]:


#convert Memory variable from string to numeric
def convert_to_gb(memory):
    #if it contains '+', take each part and convert it
    if '+' in memory:
        parts = memory.split(' + ')
        total_gb = sum(convert_to_gb(part) for part in parts)
        return total_gb

    #specifies the value and units of the string
    memory = memory.upper()
    if 'TB' in memory:
        return float(memory.split('TB')[0].replace(' ', '').replace('1.0', '1')) * 1024
    elif 'GB' in memory:
        return float(memory.split('GB')[0].replace(' ', ''))
    elif 'MB' in memory:
        return float(memory.split('MB')[0].replace(' ', '')) / 1024
    elif 'KB' in memory:
        return float(memory.split('KB')[0].replace(' ', '')) / (1024 * 1024)
    else:
        return 0  # Untuk kasus format yang tidak diketahui

#apply the conversion function to the 'Memory' column of the existing dataframe
df['Memory_GB'] = df['Memory'].apply(convert_to_gb)

#displays data from the Memory variable that has not been and has been converted to numeric
print(df[['Memory', 'Memory_GB']])


# In[664]:


#display data from Memory variables that have been converted to numeric
df['Memory_GB'].unique()


# **Encoding Process** -- Performing encoding by removing two unnecessary variables, such as the Memory and Weight variables.

# In[665]:


encoding = df.drop(columns=['Memory','Weight'])
df_encoded = pd.get_dummies(encoding, drop_first=True)
df_encoded.head()


# # Split Data (Training-Testing)

# The data splitting process divides the dataset into training and testing data. The training data (X) includes all variables except for 'Price_IDR', which is the target variable (Y). The dataset is then split into 80% for training and 20% for testing.

# In[666]:


x = df_encoded.drop(columns=['Price_IDR'])
y = df_encoded['Price_IDR']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# # Modeling

# The model used to predict laptop prices with the existing dataset is **CatBoostRegressor**. This model is highly effective for regression tasks, especially when dealing with categorical data. Additionally, CatBoostRegressor excels in delivering high performance with minimal need for hyperparameter tuning. Among several models tested, CatBoostRegressor produced the best results.

# In[667]:


#train the model
model = CatBoostRegressor(n_estimators=100, random_state=0, silent=True)
model.fit(x_train, y_train)


# In[668]:


y_pred = model.predict(x_test)


# **Model Evaluation** — The purpose of model evaluation is to determine whether the model performs well or not. In this evaluation, two metrics are used: Mean Absolute Error (MAE) and R-Squared (R²). MAE is used to indicate the average difference between the predicted and actual prices, while R² is used to assess how well the model explains the variation in laptop prices.

# In[669]:


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE : {:,.2f}".format(mae))
print("R^2 : {:.2f}".format(r2))


# # Prediction

# The prediction process is the final step for displaying the laptop price based on the desired specification.

# In[670]:


#prediction for the given laptop specification
new_laptop = pd.DataFrame({
    'Company': ['Dell'],
    'TypeName': ['Notebook'],
    'Inches': [17.3],
    'ScreenResolution': ['Full HD 1920x1080'],
    'Cpu': ['Intel Core i7 8550U 1.8GHz'],
    'Ram': [16],
    'Memory_GB': [512.0],
    'Gpu': ['AMD Radeon 530'],
    'OpSys': ['Linux Mint']
})

#encoding is done for new data (new_laptop)
new_laptop_encoded = pd.get_dummies(new_laptop, drop_first=True)
#fit the new data variable (new_laptop) to the training data and fill in the missing values with 0
new_laptop_encoded2 = new_laptop_encoded.reindex(columns=x.columns, fill_value=0)

#enter data that has been encoded and adjusted to the training data (new laptop encoded 2) into the model 
predicted_price = model.predict(new_laptop_encoded2)

print(f'Predicted Price: IDR {predicted_price[0]:,.2f}')


# # Conclusion

# The conclusion is that the predicted price for a laptop with the specified features is IDR 19,355,754.58. The difference between the predicted price and the actual price is IDR 2,216,176.67, as indicated by the MAE evaluation. This means that for a predicted price of IDR 19,355,754.58, the expected average error in the actual price is about IDR 2,216,176.67. Therefore, the actual price is likely to fall within the range of:
# 
# IDR 19,355,754.58 - IDR 2,216,176.67 to IDR 19,355,754.58 + IDR 2,216,176.67.
# 
# **Price Range : IDR 17,139,577.91 to IDR 21,571,931.25.**

# In[ ]:




