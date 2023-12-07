#!/usr/bin/env python
# coding: utf-8

# In[1]:



#Jayesh Ranjan Kesari
#getting all the libraries import which are required
import pandas as pd
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from scipy.stats import chi2_contingency, linregress
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[2]:


df = pd.read_csv('mainfile.csv')
#Original Dataset
display(df)


# In[3]:


#dropping rows which are of no use
rows_to_remove = list(range(16, 21))
df = df.drop(rows_to_remove)
display(df)


# In[4]:


#dropping columns which are of no use
df = df.drop(columns='Time Code')
df=df.drop(columns='Country Name')
#because all the value where nan and it is of no use
df=df.drop(columns='Domestic credit provided by financial sector (% of GDP) [FS.AST.DOMS.GD.ZS]')
#Surface area doesn't change and not required for visualization
df=df.drop(columns='Surface area (sq. km) [AG.SRF.TOTL.K2]')
display(df)


# In[5]:


df=df.drop(columns='Time required to start a business (days) [IC.REG.DURS]')
columns_to_remove=['Births attended by skilled health staff (% of total) [SH.STA.BRTC.ZS]','Life expectancy at birth, total (years) [SP.DYN.LE00.IN]']
df=df.drop(columns=columns_to_remove)
display(df)


# In[6]:


#Getting all the datatype of the dataset
types=df.dtypes
print(types)


# In[7]:


#changing datatype of required columns
#as year should be in int so we changing it
df['Time'] = df['Time'].astype(int)
print(df.dtypes)


# In[8]:


#unnecassary columns are removed like contraceptive one has no use in respect to GDP comparison and same with primary schooling completion rate.
columns_to_remove=['Primary completion rate, total (% of relevant age group) [SE.PRM.CMPT.ZS]','Prevalence of underweight, weight for age (% of children under 5) [SH.STA.MALN.ZS]',
                   'Poverty headcount ratio at national poverty lines (% of population) [SI.POV.NAHC]',
                   'Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population) [SI.POV.DDAY]',
                   'Income share held by lowest 20% [SI.DST.FRST.20]']
df=df.drop(columns='Contraceptive prevalence, any method (% of married women ages 15-49) [SP.DYN.CONU.ZS]')
df=df.drop(columns=columns_to_remove)


# In[9]:


mf=df
display(mf)


# In[10]:


# Defining more generic and understandable column names
new_column_names = [
    'Country_Code',
    'Time',
    'Ado_Fertility_Rate',
    'Agri_Value_Added_Percent_GDP',
    'Annual_Freshwater_Withdrawals_Percent_IR',
    'CO2_Emissions_Per_Capita',
    'Electric_Power_Consumption_kWh_Per_Capita',
    'Energy_Use_kg_Oil_Equivalent_Per_Capita',
    'Exports_Goods_Services_Percent_GDP',
    'External_Debt_Stocks_Total_Current_USD',
    'Fertility_Rate_Total_Births_Per_Woman',
    'FDI_Net_Inflows_Current_USD',
    'Forest_Area_sq_km',
    'GDP_Current_USD',
    'GDP_Growth_Annual_Percent',
    'GNI_Per_Capita_Atlas_Method_Current_USD',
    'GNI_Per_Capita_PPP_Current_International_USD',
    'GNI_Atlas_Method_Current_USD',
    'GNI_PPP_Current_International_USD',
    'Gross_Capital_Formation_Percent_GDP',
    'High_Tech_Exports_Percent_Manufactured_Exports',
    'Immunization_Measles_Percent_Children_Ages_12_23_Months',
    'Imports_Goods_Services_Percent_GDP',
    'Industry_Value_Added_Percent_GDP',
    'Inflation_GDP_Deflator_Annual_Percent',
    'Merchandise_Trade_Percent_GDP',
    'Military_Expenditure_Percent_GDP',
    'Mobile_Subscriptions_Per_100_People',
    'Mortality_Under5_Per_1000_Live_Births',
    'Net_Barter_Terms_Trade_Index_2015_100',
    'Net_Migration',
    'Net_ODA_ODA_Received_Current_USD',
    'Personal_Remittances_Received_Current_USD',
    'Population_Density_People_Per_sq_km',
    'Population_Growth_Annual_Percent',
    'Population_Total',
    'Prevalence_of_HIV_Total_Percent_Population_Ages_15_49',
    'Revenue_Excluding_Grants_Percent_GDP',
    'School_Enrollment_Primary_Percent_Gross',
    'School_Enrollment_Primary_Secondary_Gross_Gender_Parity_Index',
    'School_Enrollment_Secondary_Percent_Gross',
    'Statistical_Capacity_Score_Overall_Average_Scale_0_100',
    'Tax_Revenue_Percent_GDP',
    'Terrestrial_Marine_Protected_Areas_Percent_Total_Territorial_Area',
    'Total_Debt_Service_Percent_Exports_Goods_Services_Primary_Income',
    'Urban_Population_Growth_Annual_Percent'
]
mf.columns = new_column_names

# Set display options to show all columns
pd.set_option('display.max_columns', None)
print("\nDataFrame with more generic and understandable column names:")
display(mf)


# In[11]:


# replacing all the nan value with
mf.fillna(0, inplace=True)
display(mf)
#So that it will help in data visualization


# In[12]:


#dropping more column for more better visualization as i am comparing for gdp vs other variables factor affecting the gdps
columns_to_remove=['Terrestrial_Marine_Protected_Areas_Percent_Total_Territorial_Area',
                   'Energy_Use_kg_Oil_Equivalent_Per_Capita',
                   'Electric_Power_Consumption_kWh_Per_Capita',
                   'Annual_Freshwater_Withdrawals_Percent_IR',
                   'Prevalence_of_HIV_Total_Percent_Population_Ages_15_49',
                   'Terrestrial_Marine_Protected_Areas_Percent_Total_Territorial_Area',
                   'Prevalence_of_HIV_Total_Percent_Population_Ages_15_49',
                   'Mortality_Under5_Per_1000_Live_Births',
                   'Mobile_Subscriptions_Per_100_People',
                   'Immunization_Measles_Percent_Children_Ages_12_23_Months',
                   'Fertility_Rate_Total_Births_Per_Woman',
                   'CO2_Emissions_Per_Capita',
                   'School_Enrollment_Secondary_Percent_Gross',
                   'School_Enrollment_Primary_Secondary_Gross_Gender_Parity_Index',
                   'School_Enrollment_Primary_Percent_Gross'
                  ]
mf=mf.drop(columns=columns_to_remove)
display(mf)


# In[13]:


# Summary
display(mf.describe())


# In[14]:


#GDP TIME SERIES PLOT

# GDP over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=mf.index, y='GDP_Current_USD', color='blue')
plt.title('GDP Over Time for India')
plt.xlabel('Year')
plt.ylabel('GDP (Current USD)')
plt.show()

#GDP distribution with respect to USD(US Dollar) Over the time
plt.figure(figsize=(10, 6))
sns.barplot(x='Time', y='GDP_Current_USD', data=mf)
plt.title('Distribution of GDP Over Time')
plt.xlabel('Year')
plt.ylabel('GDP (Current USD)')
plt.show()

#GDP vs Net FDI(Foreign direct investment)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GDP_Current_USD', y='FDI_Net_Inflows_Current_USD', data=df, hue='Country_Code', alpha=0.7)
plt.title('Scatter Plot: GDP vs Foreign direct investment')
plt.xlabel('GDP (Current USD)')
plt.ylabel('Net Foreign direct investment Inflows (Current USD)')
plt.legend()
plt.show()

# Population_Growth_Annual_Percent over time 
plt.figure(figsize=(12, 6))
sns.lineplot(data=mf, x=mf.index, y='Population_Growth_Annual_Percent', color='green')
plt.title('Population_Growth_Annual_Percent')
plt.xlabel('Year')
plt.ylabel('Population Growth Rate (%)')
plt.show()

# Comparing different factors like(‘GDP_Current_USD', 'GNI_Per_Capita_Atlas_Method_Current_USD','GDP_Growth_Annual_Percent' ) how they affect.
plt.figure(figsize=(12, 6))
sns.lineplot(x='Time', y='GDP_Growth_Annual_Percent', data=df, hue=df['GDP_Growth_Annual_Percent'] > 0, palette={True: 'g', False: 'r'})
plt.title('GDP Growth Over Time with Conditional Highlighting')
plt.xlabel('Time')
plt.ylabel('GDP Growth (Annual %)')
plt.legend(title='GDP Growth > 0')
plt.show()


# In[33]:


correlation_matrix = mf.corr()

plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt='.1f', cbar=True, linewidths=.5, square=True, vmin=-1, vmax=1, annot_kws={"size": 10}, center=0)

# Set label color to teal
plt.tick_params(axis='both', colors='white')
plt.title('Correlation between different aspects', fontsize=50, color='white')

plt.xticks(rotation=45, ha='right', fontsize=50, color='white')
plt.yticks(rotation=0, ha='right', fontsize=50, color='white')

plt.show()


# In[16]:


#Military Expenditure as a Percentage of GDP (Bar Plot)
plt.figure(figsize=(10, 6))
sns.barplot(x='Time', y='Military_Expenditure_Percent_GDP', data=mf)
plt.title('Military Expenditure as a Percentage of GDP in India Over Time')
plt.xlabel('Year')
plt.ylabel('Military Expenditure (% of GDP)')
plt.show()


# In[17]:


#trying to train a model to future analysis with some main factors affecting the GDP.
#we have taken Gross annual income , annual growth percent, and exports and inports percent governing the GDP majorly.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Selecting features which you want to be predicted
features = ['GNI_Per_Capita_Atlas_Method_Current_USD', 'GDP_Growth_Annual_Percent', 'Exports_Goods_Services_Percent_GDP']
X = mf[features]
y = mf['GDP_Current_USD']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualize predictions vs. actual values
plt.figure(figsize=(10, 10))
plt.scatter(X_test['GDP_Growth_Annual_Percent'], y_test, color='blue', label='Actual')
plt.scatter(X_test['GDP_Growth_Annual_Percent'], y_pred, color='red', label='Predicted')
plt.title('GDP Prediction vs. Actual Values')
plt.xlabel('GDP_Growth_Annual_Percent')
plt.ylabel('GDP (Current USD)')
plt.legend()
plt.show()



# In[18]:


#Socio-Economic Factors

#Bar Chart of Urban Population Growth
plt.bar(df['Time'], df['Urban_Population_Growth_Annual_Percent'])
plt.xlabel('Year')
plt.ylabel('Urban Population Growth (%)')
plt.title('Bar Chart of Urban Population Growth')
plt.show()

#Line Chart of Population Total Over Time
plt.plot(df['Time'], df['Population_Total'])
plt.xlabel('Year')
plt.ylabel('Population Total')
plt.title('Line Chart of Population Total Over Time')
plt.show()

# Scatter Plot with Regression Line of Net Migration and GDP Growth
sns.regplot(x='Net_Migration', y='GDP_Growth_Annual_Percent', data=df)
plt.title('Scatter Plot with Regression Line of Net Migration and GDP Growth')
plt.show()


# Autocorrelation Plot of Population Growth
pd.plotting.autocorrelation_plot(df['Population_Growth_Annual_Percent'])
plt.title('Autocorrelation Plot of Population Growth')
plt.show()


# In[19]:


#External Factors

# Rolling Mean Plot of External Debt
df['Rolling_Mean_External_Debt'] = df['External_Debt_Stocks_Total_Current_USD'].rolling(window=3).mean()
plt.plot(df['Time'], df['External_Debt_Stocks_Total_Current_USD'], label='Original')
plt.plot(df['Time'], df['Rolling_Mean_External_Debt'], label='Rolling Mean')
plt.xlabel('Year')
plt.ylabel('External Debt (USD)')
plt.title('Rolling Mean Plot of External Debt')
plt.legend()
plt.show()

# Grouped Bar Chart of Exports and Imports
df[['Exports_Goods_Services_Percent_GDP', 'Imports_Goods_Services_Percent_GDP']].plot(kind='bar', stacked=True)
plt.xlabel('Year')
plt.ylabel('Percentage of GDP')
plt.title('Grouped Bar Chart of Exports and Imports')
plt.show()

# Bar Chart of Military Expenditure as Percentage of GDP
plt.bar(df['Time'], df['Military_Expenditure_Percent_GDP'])
plt.xlabel('Year')
plt.ylabel('Military Expenditure (% of GDP)')
plt.title('Bar Chart of Military Expenditure as Percentage of GDP')
plt.show()

# Histogram of Personal Remittances Received
plt.hist(df['Personal_Remittances_Received_Current_USD'], bins=20, edgecolor='black')
plt.xlabel('Personal Remittances Received (USD)')
plt.ylabel('Frequency')
plt.title('Histogram of Personal Remittances Received')
plt.show()


# In[ ]:




