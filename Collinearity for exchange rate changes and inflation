# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:41:56 2019

@author: bradw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Data Wrangling

#Importing data into individual data frames
#US Cpi
US_cpi = pd.read_excel('Usa cpi data.xlsx')
US_cpi.columns = US_cpi.iloc[10]
US_cpi.drop(US_cpi.index[0:11],inplace=True)
US_cpi.drop(['HALF1','HALF2'],axis=1,inplace=True)
US_cpi.set_index('Year',inplace=True)

#AU Cpi
AU_cpi = pd.read_excel('Aus cpi data.xls',sheet_name='Data1')
AU_cpi = AU_cpi.iloc[9:,1]
AU_cpi = AU_cpi.pct_change()

#Exchange Rate Data
FX_rate = pd.read_csv('AUD_USD Historical Data.csv',index_col = 'Date',parse_dates=True)
FX_rate = FX_rate.iloc[:,0]

#Combining rows into a single column
#Creating Master Data Frame
df = pd.DataFrame()

#reducing Data into time frame required
US_cpi = US_cpi.loc['1986':'2018']
Aus_CPI = AU_cpi['1986-03-01':'2018-12-01']

#FX data is value not change like other data 1 month extra aded to time frame to limit NAN value from pct change
FX_rate = pd.DataFrame(FX_rate['Dec 18': 'Sep 85'])

#Converting to quaterly values
FX_rate = FX_rate[::3]
FX_rate['FX Return'] = (FX_rate['Price'] - FX_rate['Price'].shift(-1))/ FX_rate['Price'].shift(-1) 
FX_rate = FX_rate[::-1]
FX_rate = FX_rate[2:]
FX_rate.drop('Price',axis=1,inplace=True)
FX_rate.index = Aus_CPI.index
    
#Converting monthly data to quatlerly
US_cpi['Q1'] = US_cpi[['Jan','Feb','Mar']].sum(axis=1)/100
US_cpi['Q2'] = US_cpi[['Apr','May','Jun']].sum(axis=1)/100    
US_cpi['Q3'] = US_cpi[['Jul','Aug','Sep']].sum(axis=1)/100
US_cpi['Q4'] = US_cpi[['Oct','Nov','Dec']].sum(axis=1)/100

#Creating list from data for US CPI
US_CPI = []

for i in range(len(US_cpi)):
    US_CPI.append(US_cpi['Q1'].iloc[i])
    US_CPI.append(US_cpi['Q2'].iloc[i])
    US_CPI.append(US_cpi['Q3'].iloc[i])
    US_CPI.append(US_cpi['Q4'].iloc[i])
    
Usa_CPI = pd.Series(US_CPI)


#Adding in series to make final DataFrame
df['Usa CPI'] = Aus_CPI
df['Aus CPI'] = US_CPI
df['FX Change'] = FX_rate

#Creating log prices
df['Usa CPI Log'] = np.log(df['Usa CPI'] +1)
df['Aus CPI Log'] = np.log(df['Aus CPI'] +1)
df['FX Change Log'] = np.log(df['FX Change'] +1)
df['Usa Aus Diff'] =  df['Aus CPI Log'] - df['Usa CPI Log']

#Reducing the DataFrame to just varaibles we want
df.drop(['Usa CPI','Aus CPI','FX Change'],axis=1,inplace=True)

#Data Analysis
#First step is checking for linearity of data
fig, axes = plt.subplots(nrows=3,ncols=1)
axes[0].plot(df['Usa CPI Log'])
axes[1].plot(df['Aus CPI Log'])
axes[2].plot(df['FX Change Log'])
plt.show()

#Using the AD Fuller test
ad_fuller = (stattools.adfuller(df['Usa CPI Log'],autolag ='BIC'))
print('ADF Statistic: %f' % ad_fuller[0])
print('ADF P Value: %f' % ad_fuller[1])
print('Critical Values:')
for key, value in ad_fuller[4].items():
	print('\t%s: %.3f' % (key, value))
   
#Rejecting the Null(Unit Root), data is stationary
    
#Using the AD Fuller test
ad_fuller = (stattools.adfuller(df['Aus CPI Log'],autolag ='BIC'))
print('ADF Statistic: %f' % ad_fuller[0])
print('ADF P Value: %f' % ad_fuller[1])
print('Critical Values:')
for key, value in ad_fuller[4].items():
	print('\t%s: %.3f' % (key, value))

#Rejecting the Null(Unit Root)), data is stationary

ad_fuller = (stattools.adfuller(df['FX Change Log'],autolag ='BIC'))
print('ADF Statistic: %f' % ad_fuller[0])
print('ADF P Value: %f' % ad_fuller[1])
print('Critical Values:')
for key, value in ad_fuller[4].items():
	print('\t%s: %.3f' % (key, value))
    
#Rejecting the Null(Unit Root)), data is stationary

ad_fuller = (stattools.adfuller(df['Usa Aus Diff'],autolag ='BIC'))
print('ADF Statistic: %f' % ad_fuller[0])
print('ADF P Value: %f' % ad_fuller[1])
print('Critical Values:')
for key, value in ad_fuller[4].items():
	print('\t%s: %.3f' % (key, value))
    
    
plot_acf(df['Usa CPI Log'], lags=10)
plot_acf(df['Aus CPI Log'], lags=10)
plot_acf(df['FX Change Log'], lags=10)
plot_acf(df['Usa Aus Diff'], lags=10)

plot_pacf(df['Usa CPI Log'], lags=10)
plot_pacf(df['Aus CPI Log'], lags=10)
plot_pacf(df['FX Change Log'], lags=10)
plot_pacf(df['Usa Aus Diff'], lags=10)

#Adding in t-1 and t-2 of USA CPI data variables into the model and checking the outputs. Mean reversion of ACF and 2 lag PACF

#Creating Variables
df['Usa CPI Log T-1'] = df['Usa CPI Log'].shift(1)
df['Usa CPI Log T-2'] = df['Usa CPI Log'].shift(2)

#Removing NAN values from data set
df = df[pd.notnull(df['Usa CPI Log T-2'])]


#Checking significance of ar1 and ar2 values
#Creating a model
y = df['Usa CPI Log']
X = df[['Usa CPI Log T-1','Usa CPI Log T-2']]

#Fitting Model
results = sm.OLS(y,X).fit()
print(results.summary())

#Both significant so including in model

#Creating a model
y = df['FX Change Log']
X = df.drop(['Aus CPI Log','FX Change Log','Usa CPI Log'],axis=1)

#Fitting Model
results = sm.OLS(y,X).fit()
print(results.summary())

#Checking for AC and PAC in model to make sure all is taken out with US CPI Log t-1 t-2
residuals = results.resid
plt.hist(residuals)
plt.show()
#Residuals seem to be normally distributed around a mean of 0

plot_acf(residuals, lags=10)
plot_pacf(residuals, lags=10)

#No auto correlation in the model 
ad_fuller = (stattools.adfuller(residuals,autolag ='BIC'))
print('ADF Statistic: %f' % ad_fuller[0])
print('ADF P Value: %f' % ad_fuller[1])
print('Critical Values:')
for key, value in ad_fuller[4].items():
	print('\t%s: %.3f' % (key, value))
    
#T stat outside of critical value for Engle Granger test for collinearity of 3.37 (T Stat -10.92)
#Reject the null, data has no collinearity therefore PPP does not hold from this sample.



