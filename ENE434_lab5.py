# ENE434 LAB5 - Forecasting methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa._stl import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import pmdarima as pm
from scipy import stats
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import os

# Load ets data
ets = pd.read_csv("https://raw.githubusercontent.com/ysture/ENE434/master/Data/eua-price.csv")
ets.Price = (ets.Date.astype(str) + '.' + ets.Price.astype(str)).astype(float)
ets.Date = ets.index
ets.reset_index(drop=True, inplace=True)
ets.Date = pd.to_datetime(ets.Date, format='%Y-%m-%d')
ets.columns = ['date', 'price']

# Aggregating on month (mean of price each month)
ets['month'] = [x.month for x in ets.date]
ets['year'] = [x.year for x in ets.date]

ets_mon = ets.groupby(by=['year', 'month']).mean()
ets_mon['day'] = 1
ets_mon.reset_index(inplace=True)
ets_mon['date'] = pd.to_datetime(ets_mon.year.astype(str) + '-' + ets_mon.month.astype(str) + '-' + ets_mon.day.astype(str))
ets_mon

# Load elspot data
elspot = pd.read_csv("http://jmaurit.github.io/norwayeconomy/data_series/elspot.csv")
elspot.date = pd.to_datetime(elspot.date)

# Merge data frames
power_df = elspot.merge(ets_mon[['date','price']], on='date', how='inner')

# Write power_df to .csv
power_df.to_csv("power_df.csv")

# Start by investigating Denmark
dk_df = power_df[['date','DK1', 'DK2','price']].copy()
dk_df.columns = ["date", "DK1_price", "DK2_price", "ets_price"]

# Convert DK1_price and DK2_price to thousands
dk_df.DK1_price = dk_df.DK1_price/1000
dk_df.DK2_price = dk_df.DK2_price/1000

# Plot
fig, ax = plt.subplots()
ax = plt.subplot(1,1,1)

for i in range(1,4):
    col = dk_df.columns[i]
    plt.plot(dk_df['date'], dk_df[col], label=col)

plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#### Forecasting ARIMA models

# Creating new df of just dk1_price and date
dk1price_ts = dk_df[['date', 'DK1_price']]

# ADF-test for stationarity. # NOT same as in lab (he has p-value of 0.04469)
adf = adfuller(dk1price_ts['DK1_price'])
print('p-value: {}'.format(adf[1]))

# Looks like the same ts as in class
plt.plot(dk1price_ts['date'], dk1price_ts['DK1_price'])
plt.show()

d_dk1price_ts = dk1price_ts.diff()
d_adf = adfuller(d_dk1price_ts['DK1_price'].dropna())
print('p-value: {}'.format(d_adf[1]))

# Both look like the same as in class
plt.figure(figsize=(8,6))
fig, axes = plt.subplots(1, 2)
plot_acf(d_dk1price_ts['DK1_price'].dropna(), ax=axes[0])
plot_pacf(d_dk1price_ts['DK1_price'].dropna(), ax=axes[1], zero=False)
plt.show()

# Creating ARIMA models
fit1 = ARIMA(dk1price_ts['DK1_price'], order=(0,1,1))
fit2 = ARIMA(dk1price_ts['DK1_price'], order=(1,1,0))
res1 = fit1.fit()
res2 = fit2.fit()

res1.summary() # MA Coefficient approximately same as in class
res2.summary() # MA Coefficient same as in class

def residual_plot(arima_mod):
    result = arima_mod.fit()
    residuals = result.resid
    # Creating residual plot (similar to the one in class)
    fig = plt.figure(figsize=(12,8))
    #fig.subplots_adjust(wspace=0.05)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(residuals)
    order = arima_mod.order
    seasorder = arima_mod.seasonal_order
    title = ('Residuals from ARIMA{}{}'.format(order, seasorder))
    ax1.set_title(title, fontdict={'size':20}, loc='left')
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    plot_acf(residuals, ax=ax2, zero=False)
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax3.hist(residuals)
    plt.show()
    del result
    del arima_mod

fit1 = ARIMA(dk1price_ts['DK1_price'], order=(0,1,1))
fit2 = ARIMA(dk1price_ts['DK1_price'], order=(1,1,0))
residual_plot(fit1)
residual_plot(fit2)

# Ljung-Box test says that there is no autocorrelation in the residuals
lj = acorr_ljungbox(res1.resid, lags=10)
lj[1]

# Trying AR(2) model. "Differencing doesnt come free. You tend to throw away a lot of information, and this will also impact the forecast you end up making, leading to more uncertainty."
fit3 = ARIMA(dk1price_ts['DK1_price'], order=(2,0,0))
res3 = fit3.fit()

# Approximately same as in class
fit3 = ARIMA(dk1price_ts['DK1_price'], order=(2,0,0))
residual_plot(fit3)
res3.summary()

lj = acorr_ljungbox(res3.resid, lags=10)
lj[1]

def forecast_plot(dataframe, arima_mod, forecasts=12, outer_interval=0.95, inner_interval=0.8):
    # Forecast ("The result of the forecast() function is an array containing the forecast value, the standard error of the forecast, and the confidence interval information.")
    res = arima_mod.fit()

    f = res.get_forecast(forecasts).summary_frame()
    # Forecast index
    f_ix = f.index
    # Prediction mean
    pred_mean = f['mean']
    # Confidence intervals
    under_outer = res.get_forecast(forecasts).conf_int(alpha=1-outer_interval).iloc[:,0]
    over_outer = res.get_forecast(forecasts).conf_int(alpha=1-outer_interval).iloc[:,1]
    under_inner = res.get_forecast(forecasts).conf_int(alpha=1-inner_interval).iloc[:,0]
    over_inner = res.get_forecast(forecasts).conf_int(alpha=1-inner_interval).iloc[:,1]

    # res3.predict()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(dataframe.index, dataframe, color='black', label='Training set')
    ax.plot(f_ix, pred_mean, color='blue', label='Forecast')
    # Confidence intervals
    plt.fill_between(f_ix, under_outer, over_outer, color='b', alpha=0.2, label='{}% confidence interval'.format(int(outer_interval*100)))
    plt.fill_between(f_ix, under_inner, over_inner, color='b', alpha=0.4, label='{}% confidence interval'.format(int(inner_interval*100)))


    order = arima_mod.order
    seas_order = arima_mod.seasonal_order
    plt.title('Forecasts from ARIMA{}{}'.format(order, seas_order),
              fontdict={'size':19}, loc='left')
    plt.legend(loc='upper center')
    plt.show()

'''
fit1 = ARIMA(dk1price_ts['DK1_price'], order=(0,1,1))
forecast_plot(fit1)
fit3 = ARIMA(dk1price_ts['DK1_price'], order=(2,0,0))
forecast_plot(fit3)
'''

## Seasonality forecasting
cons = pd.read_csv("http://jmaurit.github.io/analytics/labs/data/consumption-per-country_2019_daily.csv", delimiter=';')
cons.date = pd.to_datetime(cons.date, format='%d/%m/%Y')


cons_no = cons[['NO']]
cons_no.index = cons['date']
cons_no = cons_no.asfreq('d')
# STL decomposition
stl = STL(cons_no, period=7, seasonal=51)
res = stl.fit()
res.plot()
plt.show()

# Seasonal ARIMA forecasting
sfit1 = ARIMA(cons_no, order=(1,0,1), seasonal_order = (0,1,1,7))
res = sfit1.fit(method='innovations_mle')
res.summary()

'''
# Plot residuals. RESIDUAL PLOTS LOOK STRANGE
residual_plot(sfit1)

'''
# Plot forecast
fig = plt.figure()
plt.plot(cons_no, color='black')
plt.plot(res.forecast(30), color='blue')
plt.show()

#### Assignment exercises
### 1)
cons_dk = cons[['DK']]
cons_dk.index = cons['date']
cons_dk = cons_dk.asfreq('d')


## 1.2 Comparing Norwegian and Danish consumption
fig = plt.figure()
plt.plot(cons_dk, label='Denmark', color='Red')
plt.plot(cons_no, label='Norway', color='Blue')
plt.title('2019 Energy Consumption in Denmark and Norway')
plt.legend(loc='best')
plt.show()


## 1.2 Forecasting Danish consumption
# Creating differenced and seasonal differenced time series
cons_dk_diff = cons_dk.diff()
cons_dk_seasdiff = cons_dk.diff(7)

# Arima Forecasting
cons_dk.plot()
plt.title('DK Consumption 2019')
plt.show()

cons_dk_diff.plot()
plt.title('DK Consumption 2019 \n First order difference')
plt.show()

cons_dk_seasdiff.plot()
plt.title('DK Consumption 2019 \n First order seasonal difference')
plt.show()


# ADFuller original timeseries
print('P-value of ADFuller test: {:.6f}'.format(adfuller(cons_dk)[1]))
# ADFuller first-order differenced timeseries
print('P-value of ADFuller test: {:.6f}'.format(adfuller(cons_dk_diff.dropna())[1]))
# ADFuller first-order seasonally differenced timeseries
print('P-value of ADFuller test: {:.6f}'.format(adfuller(cons_dk_seasdiff.dropna())[1]))

# Plotting ACF
fig, axes = plt.subplots(2,3, sharex=True)
plot_acf(cons_dk, ax=axes[0,0], title='Autocorrelaction \n Ordinary')
plot_pacf(cons_dk, ax=axes[1,0], title='Partial \n Ordinary')
plot_acf(cons_dk_diff.dropna(), ax=axes[0,1], title='Autocorrelaction \n Differenced (1)')
plot_pacf(cons_dk_diff.dropna(), ax=axes[1,1], title='Partial \n Difference (1)')
plot_acf(cons_dk_seasdiff.dropna(), ax=axes[0,2], title='Autocorrelaction \n Seasonal (1)')
plot_pacf(cons_dk_seasdiff.dropna(), ax=axes[1,2], title='Partial \n Seasonal (1)')
plt.show()

# ADF and partial autocorrelation plots suggest that seasonal differencing is best (ADF suggest normal, but pacf plots otherwise)
# End up with ARIMA(1,0,2)(2,1,0)[7] as the best model, as there are two significant seasonal lags in the
# pacf plot as well as three significant lags in the ACF plot.

# Creating forecast
sfit1 = ARIMA(cons_dk, order=(1,0,2), seasonal_order = (2,1,0,7))
res = sfit1.fit(method='innovations_mle')
res.predict()
res2 = sfit1.fit(transformed=False)
res.summary()

# Model diagnostics
res.plot_diagnostics(lags=25)
plt.show()

# Trying to get fitted values to correspond to R's fitted values
fig = plt.figure()
observations_skipped = 10
predictions = res.predict(dynamic=False)[observations_skipped:]
plt.plot(cons_dk.index[observations_skipped:], cons_dk[observations_skipped:], label='ts')
plt.plot(predictions.index, predictions, label='Fitted values', linestyle="--")
plt.legend(loc='best')
plt.show()

# Plotting forecast
sfit1 = ARIMA(cons_dk, order=(1,0,2), seasonal_order = (2,1,0,7))
forecast_plot(cons_dk, sfit1)


### 2) Choose prices for a certain country for 2019. Model the dynamics of power prices, including for checking for and modeling conditional variance. Create a forecast for 30 days.
## Daily elspot prices in Lativa for 2019
eur = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Data/elspot-prices_2019_daily_eur.csv', encoding='unicode_escape')
ix = eur['Unnamed: 0'].str.replace('/','-')
eur.index = pd.to_datetime(ix, format='%d-%m-%Y')
lt = eur[['LT']]

'''
# define ARCH model
model = arch_model(lt, mean='Zero', vol='GARCH', p=15)
model.fit(update_freq = 5)
'''
