'''
Script with final, individual ENE434 Assignment.
The script investigates the relation between Purchasing Managers' Index (PMI), oil price and national electricity
prices in Norway, Denmark, UK and US.
'''
import pandas as pd
import numpy as np
import math
from tabula import read_pdf
import os
import matplotlib.pyplot as plt
from datetime import datetime, date
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
import matplotlib.lines as mlines
from pmdarima.arima.utils import ndiffs, nsdiffs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix, accuracy_score, mean_squared_error
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Importing currency exchange rates
cur = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/macrobond_currency.csv',
                  header=0,
                  names=['month', 'eur_per_usd', 'gbp_per_usd'])
cur['month'] = pd.to_datetime(cur['month'])
currency_dict = {'eur':'eur_per_usd', 'gbp':'gbp_per_usd'}

# Helper functions
# To import data from NordPool
def import_nordpool():
    df = pd.DataFrame()
    for f in nordpool_files:
        dfTemp = pd.read_csv('input/{}'.format(f), engine='python', header=2)
        df = pd.concat([df, dfTemp], axis=0)
    df = df.reset_index(drop=True).rename(columns={'Unnamed: 0':'month'})
    df = df.stack().str.replace(',','.').unstack()
    return df

# Import US electricity data
def import_us():
    df = pd.DataFrame()
    for f in us_filenames:
        dfTemp = pd.read_csv('input/ice_electric-historical/{}'.format(f), engine='python')
        dfTemp.columns = dfTemp.columns.str.lstrip()
        dfTemp.columns = dfTemp.columns.str.rstrip()
        dfTemp = dfTemp[['Price Hub', 'Trade Date', 'Wtd Avg Price $/MWh', 'Daily Volume MWh']]
        dfTemp['Daily Volume MWh'] = dfTemp['Daily Volume MWh'].str.replace(" ", "")
        df = pd.concat([df, dfTemp], axis=0)
    df = df.astype({'Wtd Avg Price $/MWh': 'float', 'Daily Volume MWh': 'int'})
    df = df.reset_index(drop=True)
    return df

# To remove projections from certain PMI tables
def remove_projections(df):
    df['monthyear'] = df.month.dt.strftime('%m-%Y')
    for x in df.monthyear.unique():
        if sum(df.monthyear == x) > 1:
            df.drop(df.month[df. month == max(df.month[df.monthyear == x])].index, axis=0, inplace=True)
    return df

# To convert electricity prices to USD
def convert_currency(df):
    for abbr in list(currency_dict.keys()):
        if df.columns[1].startswith(abbr):
            for index, row in df.iterrows():
                convert_to = currency_dict[abbr]
                period = np.datetime64(row.month)
                try:
                    exchange_rate = cur[convert_to][cur.month.isin([period])].values[0]
                except IndexError:
                    exchange_rate = np.nan
                original_price = row[1]
                df.loc[index, 'usd_per_MWh'] = original_price/exchange_rate
    return df

# To convert column types of PMI data frames
def convert_dtypes(df):
    df = df.astype({'pmi':'float'})
    df['month'] = pd.to_datetime(df['month'])
    return df

# Making date the index for all dataframes
def month_as_index(df):
    df.index = df.month
    df.drop(['month'], inplace=True, axis=1)
    return df

# Shifting month one month back (for UK and Germany PMI data)
def shift_month_back(df):
    shifted_back_list = []
    for r in df.month:
        m = r.month
        y = r.year
        if len(str(m)) < 2 or m==10:
            m_new = '0{}'.format(m-1)
        else:
            m_new = m-1
        if m == 1:
            y_new = y-1
        else:
            y_new = y
        if m_new == '00':
            m_new = 12
        new_date = np.datetime64('{}-{}'.format(y_new, m_new))
        shifted_back_list.append(new_date)
    return shifted_back_list

# Function to plot decomposition and the different decomposed elements
def plot_decomposition(df, column_index, plot_title):
    try:
        df = df.dropna()
        value_column = df.iloc[:, column_index]
        decomp = seasonal_decompose(value_column, period=12)

        fig, axes = plt.subplots(nrows=4, ncols=1)

        ax = axes[0]
        ax.plot(df.month, value_column)
        ax = axes[1]
        ax.plot(df.month, decomp.trend)
        ax.set_title('Trend')
        ax = axes[2]
        ax.plot(df.month, decomp.seasonal)
        ax.set_title('Seasonality')
        ax = axes[3]
        ax.plot(df.month, decomp.resid)
        ax.set_title('Residuals')

        plt.suptitle(plot_title, y=1)

        #plt.tight_layout()
        plt.show()
    except:
        raise Exception('Could not plot {}'.format(plot_title))

# Function to plot autocorrelation
def plot_autocorrelation(dict, column_index, filename, suptitle):
    print('ADFuller test to check for stationarity (H0 is that there is non-stationarity):')
    for i in range(len(list(dict.values()))):
        df = list(dict.values())[i].dropna()
        p_val = adfuller(df.iloc[:, column_index])[1] # ADFuller test
        ndiff = ndiffs(df.iloc[:,column_index], test='adf')

        title = list(dict.keys())[i]

        plot_pacf(df.iloc[:,column_index], ax=axes[0,i], title=title)
        axes[0,i].text(x=4, y=0.85, s='ADFuller: {}'.format(round(p_val,4)), fontdict={'color':'#8b0000'})
        axes[0,i].text(x=4, y=0.65, s='Ndiffs: {}'.format(ndiff), fontdict={'color':'black'})
        plot_acf(df.iloc[:,column_index], ax=axes[1,i], title=title)

        # Print ADFuller test
        print('P-value of {c}: {p}'.format(c=title, p=p_val))
    plt.suptitle(suptitle, fontweight='bold')
    #fig.align_ylabels()
    plt.savefig('plots/{}.png'.format(filename))
    plt.show()

# Function to plot forecasts
def forecast_plot(dataframe, arima_mod, forecasts=12, outer_interval=0.95, inner_interval=0.8, y_label="", x_label="", exog_test=None):

    res = arima_mod.fit(maxiter=100, disp=0)

    # Forecast ("The result of the forecast() function is an array containing the forecast value, the standard error of the forecast, and the confidence interval information.")
    if exog_test is not None:
        f = res.get_forecast(forecasts, exog=exog_test.iloc[-forecasts:,:].astype('float')).summary_frame()

        # Forecast index
        f_ix = f.index
        # Prediction mean
        pred_mean = f['mean']
        # Confidence intervals
        under_outer = res.get_forecast(forecasts, exog=exog_test.iloc[-forecasts:,:].astype('float')).conf_int(alpha=1-outer_interval).iloc[:,0]
        over_outer = res.get_forecast(forecasts , exog=exog_test.iloc[-forecasts:,:].astype('float')).conf_int(alpha=1-outer_interval).iloc[:,1]
        under_inner = res.get_forecast(forecasts, exog=exog_test.iloc[-forecasts:,:].astype('float')).conf_int(alpha=1-inner_interval).iloc[:,0]
        over_inner = res.get_forecast(forecasts , exog=exog_test.iloc[-forecasts:,:].astype('float')).conf_int(alpha=1-inner_interval).iloc[:,1]
    else:
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
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(dataframe.index, dataframe, color='black', label='Training set')
    ax.plot(f_ix, pred_mean, color='blue', label='Forecast')
    # Confidence intervals
    #plt.fill_between(f_ix, under_outer, over_outer, color='b', alpha=0.2, label='{}% confidence interval'.format(int(outer_interval*100)))
    #plt.fill_between(f_ix, under_inner, over_inner, color='b', alpha=0.4, label='{}% confidence interval'.format(int(inner_interval*100)))


    order = arima_mod.order
    seas_order = arima_mod.seasonal_order
    plt.title('Forecasts from ARIMA{}{}'.format(order, seas_order),
              fontdict={'size':19}, loc='left')
    plt.legend(loc='upper center')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

def plot_multiple_forecast(res, res_wd, train_set, test_set, exog_test, title, ax):
    n_test_obs = len(test_set)
    preds = res.get_forecast(n_test_obs, exog=exog_test.astype('float')).summary_frame()['mean']
    preds_wd = res_wd.get_forecast(n_test_obs, exog=exog_test.astype('float')).summary_frame()['mean']
    naive = [train_set[-1]]*len(preds)

    # Plotting
    ax.plot(train_set, label='Training set', color='black')
    ax.plot(preds, label='Forecast (d=1)', color='blue')
    ax.plot(preds_wd, label='Forecast (d=0)', color='red')
    ax.plot(test_set, label='Test set', color='orange', linestyle='--')
    ax.plot(preds.index, naive, label='Naïve', color='gray', linestyle='--')
    ax.set_ylabel('PMI')
    ax.set_title(title, fontsize=14)


### PMI
## Norway
# Ordinary
pmi_no_list = read_pdf(
    'C:\\Users\\Yngve\\Google Drive\\Skolerelatert\\NHH\\Master\\ENE434\\Final assignment\\input\\PMI_norway_pdf.pdf',
    pages=[1, 2, 3, 4, 5], multiple_tables=False,
    pandas_options={'header': None, 'usecols':[0,1], 'names': ['month', 'pmi']})

# Seasonally adjusted
pmi_no_seasadj_list = read_pdf(
    'C:\\Users\\Yngve\\Google Drive\\Skolerelatert\\NHH\\Master\\ENE434\\Final assignment\\input\\PMI_norway_pdf.pdf',
    pages=[6, 7, 8, 9, 10], multiple_tables=False,
    pandas_options={'header': None, 'usecols':[0,1], 'names': ['month', 'pmi']})


pmi_no = pmi_no_list[0].dropna()
pmi_no['month'] = pd.to_datetime(pmi_no['month'], dayfirst=True)
pmi_no_seasadj = pmi_no_seasadj_list[0].dropna()
pmi_no_seasadj['month'] = pd.to_datetime(pmi_no_seasadj['month'], dayfirst=True)

# UK
pmi_uk = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/united-kingdom.markit-manufacturing-pmi.csv',
                     sep='\t',
                     usecols=[0,1],
                     header=0,
                     names=['month', 'pmi'])
pmi_uk['month'] = pd.to_datetime(pmi_uk['month'], dayfirst=True)
pmi_uk = remove_projections(pmi_uk)
pmi_uk['month'] = shift_month_back(pmi_uk)


## Denmark and US
pmi_udk = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/us_dk_pmi.csv',
                      sep=';',
                      header=0,
                      names=['month', 'us_index', 'dk_index'])

pmi_udk = pmi_udk.stack().str.replace(',','.').unstack()

# US
pmi_us = pmi_udk[['month', 'us_index']].rename(columns={'us_index':'pmi'})
pmi_us['month'] = pd.to_datetime(pmi_us['month'], dayfirst=True)

# Denmark
pmi_dk = pmi_udk[['month', 'dk_index']].dropna().reset_index(drop=True).rename(columns={'dk_index':'pmi'})
pmi_dk['month'] = pd.to_datetime(pmi_dk['month'], dayfirst=True)

### Electricity
# UK
el_uk = pd.read_csv(
    'https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/UK_Electricityprices_Day_monthlyaverage.csv',
    header=0,
    names=['month', 'gbp_per_MWh'])
el_uk['month'] = pd.to_datetime(el_uk['month'], dayfirst=True)

## NordPool (Norway and Denmark)
nordpool_files = [f for f in os.listdir('input/') if f.startswith('elspot-prices_')]
np_df = import_nordpool()
np_df.columns
np_df['month'] = np_df['month'].apply(lambda x: datetime.strptime(x, '%y - %b'))

# Norway
el_no = np_df[['Oslo', 'Kr.sand', 'Bergen', 'Molde', 'Tr.heim', 'Tromsø']].astype('float')
el_no = pd.concat([np_df.month, el_no.mean(axis=1)], axis=1)
el_no.columns = ['month', 'eur_per_MWh']
el_no = convert_currency(el_no)
el_no.dropna(inplace=True)

# Denmark
el_dk = np_df[['DK1', 'DK2']].astype('float')
el_dk = pd.concat([np_df.month, el_dk.mean(axis=1)], axis=1)
el_dk.columns = ['month', 'eur_per_MWh']
el_dk.dropna(inplace=True)

# US
us_filenames = os.listdir('input/ice_electric-historical')
us_df = import_us()

us_df['Trade Date'] = pd.to_datetime(us_df['Trade Date'])
us_df.sort_values(by='Trade Date')

# Calculating weighted Average
g = us_df.groupby(by='Trade Date').agg({'Daily Volume MWh': 'sum'}).rename(
    columns={'Daily Volume MWh': 'daily_vol'}).reset_index()
merged = pd.merge(us_df, g, on='Trade Date')
weights = merged['Daily Volume MWh'] / merged['daily_vol']
merged['weighted_avg'] = merged['Wtd Avg Price $/MWh'] * weights
merged = merged.groupby(by='Trade Date'). \
    agg({'weighted_avg': 'sum', 'daily_vol': 'mean'}).reset_index().rename(columns={'Trade Date': 'month',
                                                                                    'weighted_avg': 'usd_per_MWh'})
merged['month_year'] = merged.month.dt.strftime('%m-%Y')
j = merged.groupby(pd.Grouper(key='month', freq='M')).agg({'daily_vol': 'sum'}).rename(
    columns={'daily_vol': 'monthly_vol'}).reset_index()
j['month_year'] = j.month.dt.strftime('%m-%Y')
merged = pd.merge(merged, j, on='month_year', how='left')
weights = merged['daily_vol'] / merged['monthly_vol']
merged['weighted_avg_monthly_usd'] = merged['usd_per_MWh'] * weights

el_us = merged.groupby(by='month_year'). \
    agg({'weighted_avg_monthly_usd': 'sum'}).reset_index().rename(columns={'month_year': 'month',
                                                                           'weighted_avg_monthly_usd': 'usd_per_MWh'})
el_us['month'] = pd.to_datetime(el_us.month, dayfirst=False)
el_us.sort_values(by='month', inplace=True)

# Convert electricity prices to USD
el_uk = convert_currency(el_uk)
el_no = convert_currency(el_no)
el_dk = convert_currency(el_dk)

### Oil and gas
# WTI
wti = pd.read_csv(
    'https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/Cushing_OK_WTI_Spot_Price_FOB.csv',
    skiprows=4,
    header=0,
    names=['month', 'usd_per_barrel'])
wti['month'] = pd.to_datetime(wti.month)

# Brent
brent = pd.read_csv(
    'https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/Europe_Brent_Spot_Price_FOB.csv',
    skiprows=4,
    header=0,
    names=['month', 'usd_per_barrel'])
brent['month'] = pd.to_datetime(brent.month)

# Delete variables in environment not needed in the further analysis
del([merged, pmi_no_list, pmi_udk, us_filenames, weights, g, np_df, us_df, nordpool_files, pmi_no_seasadj_list])

# Convert column types
pmi_no = convert_dtypes(pmi_no)
pmi_no_seasadj = convert_dtypes(pmi_no_seasadj)
pmi_dk = convert_dtypes(pmi_dk)
pmi_uk = convert_dtypes(pmi_uk)
pmi_us = convert_dtypes(pmi_us)


'''
Descriptive statistics in this order:
1. PMI
2. Electricity prices
3. Oil prices
'''
#### Decomposed series
### PMI
pmi_dk_decomp = seasonal_decompose(pmi_dk.pmi, period=12).trend.dropna()
pmi_dk_seasadj = pd.DataFrame({'month':pmi_dk.month[pmi_dk.index.isin(pmi_dk_decomp.index)],
                               'pmi':pmi_dk_decomp})


pmi_dict = {'Norway':pmi_no,
            'Denmark':pmi_dk,
            'UK':pmi_uk,
            'US':pmi_us}


# Each PMI series individually
fig, ax = plt.subplots(ncols=1, nrows=len(pmi_dict.keys()), figsize=(6,8))
i=0
plt.subplots_adjust(hspace=0.5)
for row in ax:
    df = list(pmi_dict.values())[i]
    row.plot(df['month'], df['pmi'])
    title = list(pmi_dict.keys())[i] + ' PMI'
    row.set_title(title)
    row.tick_params(axis='x', labelrotation=0)
    # Next iteration
    i+=1
plt.savefig('plots/pmi_from_start.png')
plt.show()


# In a single plot (from 2008)
start_date = np.datetime64(date(2008, 1, 1))
fig, ax = plt.subplots(figsize=(8,5))
i=0
for i in range(len(pmi_dict.values())):
    df = list(pmi_dict.values())[i]
    df = df[df['month']> start_date]
    label = list(pmi_dict.keys())[i]
    ax.plot(df['month'], df['pmi'], label=label)
    ax.set_title('PMI 2008-2020', fontdict={'size':18})
plt.legend(loc='best')
plt.savefig('plots/pmi_from_08.png')
plt.show()



### Electricity prices
el_no = el_no[['month', 'usd_per_MWh', 'eur_per_MWh']]
el_uk = el_uk[['month', 'usd_per_MWh', 'gbp_per_MWh']]
el_us = el_us[['month', 'usd_per_MWh']]
el_dk = el_dk[['month', 'usd_per_MWh', 'eur_per_MWh']]

el_dict = {'Norway':el_no,
           'Denmark':el_dk,
           'UK':el_uk,
           'US':el_us}
# In a single plot
start_date = np.datetime64(date(2005, 1, 1))
fig, ax = plt.subplots(figsize=(8,5))
i=0
for i in range(len(el_dict.values())):
    df = list(el_dict.values())[i]
    df = df[df['month']> start_date]
    df.dropna(inplace=True)
    label = list(el_dict.keys())[i]
    ax.plot(df['month'], df['usd_per_MWh'], label=label)
    ax.set_title('$/MWh 2005-2020', fontdict={'size':18})
plt.legend(loc='upper right')
plt.savefig('plots/el_from_04.png')
plt.show()


### Oil prices
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(brent.month, brent.usd_per_barrel, label='Brent')
ax.plot(wti.month, wti.usd_per_barrel, label='WTI')
ax.set_title('$/Barrel 1986-2020', fontdict={'size':18})
plt.legend(loc='best')
plt.savefig('plots/oil.png')
plt.show()


'''
Investigating seasonality for all time series, should any of the series be seasonally adjusted?
1. PMI
2. Electricity
3. Oil prices
'''

# PMI
plot_decomposition(pmi_dk, 1, 'Denmark PMI')
plot_decomposition(pmi_no, 1, 'Norway PMI')
plot_decomposition(pmi_us, 1, 'US PMI')
plot_decomposition(pmi_uk, 1, 'UK PMI')

# Electricity
plot_decomposition(el_dk, 2, 'Electricity Denmark')
plot_decomposition(el_no, 2, 'Electricity Norway')
plot_decomposition(el_uk, 2, 'Electricity UK')
plot_decomposition(el_us, 1, 'Electricity US')

# Oil
plot_decomposition(brent, 1, 'Brent Oil')
plot_decomposition(wti, 1, 'WTI Oil')


'''
Investigating autocorrelation for all time series, should any of the series be seasonally adjusted?
1. PMI
2. Electricity
3. Oil prices

# Plotting ACF and PACF
'''
# PMI
fig, axes = plt.subplots(ncols=len(pmi_dict.keys()), nrows=2, figsize=(10,5))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
title_size = 15
title_type = 'bold'
ylabel_size = 12

plot_autocorrelation(pmi_dict, 1, 'pmi_pacf', 'PMI')

# Electricity
fig, axes = plt.subplots(ncols=len(el_dict.keys()), nrows=2, figsize=(10,5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
title_size = 15
title_type = 'bold'
ylabel_size = 12
plot_autocorrelation(el_dict, 1, 'el_pacf', 'Electricity')

# Oil
oil_dict = {'Brent':brent,
            'WTI': wti}

fig, axes = plt.subplots(nrows=2, ncols=len(oil_dict.keys()), figsize=(7,5))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
title_size = 15
title_type = 'bold'
ylabel_size = 12
plot_autocorrelation(oil_dict, 1, 'oil_pacf', 'Oil')

'''
Preparing data for modelling
'''

# Preparing data for modelling
# Creating lagged series
brent['usd_per_barrel'] = brent['usd_per_barrel']
wti['usd_per_barrel'] = wti['usd_per_barrel']
brent['brent_lag_1'] = brent['usd_per_barrel'].shift(-1)
brent['brent_lag_2'] = brent['usd_per_barrel'].shift(-2)
wti['wti_lag_1'] = wti['usd_per_barrel'].shift(-1)
wti['wti_lag_2'] = wti['usd_per_barrel'].shift(-2)

el_no['el_lag_1'] = el_no['usd_per_MWh'].shift(-1)
el_no['el_lag_2'] = el_no['usd_per_MWh'].shift(-2)
el_dk['el_lag_1'] = el_dk['usd_per_MWh'].shift(-1)
el_dk['el_lag_2'] = el_dk['usd_per_MWh'].shift(-2)
el_uk['el_lag_1'] = el_uk['usd_per_MWh'].shift(-1)
el_uk['el_lag_2'] = el_uk['usd_per_MWh'].shift(-2)
el_us['el_lag_1'] = el_us['usd_per_MWh'].shift(-1)
el_us['el_lag_2'] = el_us['usd_per_MWh'].shift(-2)

# Making data for all countries the same length. The length of the time series
# is decided by the shortest time series length for each country
pmi_no = pmi_no[pmi_no.month.isin(el_no.month)]
pmi_dk = pmi_dk[pmi_dk.month.isin(el_dk.month)]
pmi_uk = pmi_uk[pmi_uk.month.isin(el_uk.month)]
pmi_us = pmi_us[pmi_us.month.isin(el_us.month)]

# Making date index for all dataframes
pmi_no = month_as_index(pmi_no)
pmi_dk = month_as_index(pmi_dk)
pmi_us = month_as_index(pmi_us)
pmi_uk = month_as_index(pmi_uk)

el_dk = month_as_index(el_dk)
el_no = month_as_index(el_no)
el_uk = month_as_index(el_uk)
el_us = month_as_index(el_us)

brent = month_as_index(brent)
wti = month_as_index(wti)

# Creating one dataframe for each country to include exogenous variables and PMI in the same df
df_no = pd.merge(pmi_no, el_no, how='left', left_index=True, right_index=True)
df_no = pd.merge(df_no, brent, how='left', left_index=True, right_index=True)
df_no = pd.merge(df_no, wti, how='left', left_index=True, right_index=True)
df_no = df_no.dropna()

# Creating one dataframe for each country to include exogenous variables and PMI in the same df
df_dk = pd.merge(pmi_dk, el_dk, how='left', left_index=True, right_index=True)
df_dk = pd.merge(df_dk, brent, how='left', left_index=True, right_index=True)
df_dk = pd.merge(df_dk, wti, how='left', left_index=True, right_index=True)
df_dk = df_dk.dropna()

# Creating one dataframe for each country to include exogenous variables and PMI in the same df
df_uk = pd.merge(pmi_uk, el_uk, how='left', left_index=True, right_index=True)
df_uk = pd.merge(df_uk, brent, how='left', left_index=True, right_index=True)
df_uk = pd.merge(df_uk, wti, how='left', left_index=True, right_index=True)
df_uk = df_uk.dropna()
df_uk.sort_index(inplace=True)

# Creating one dataframe for each country to include exogenous variables and PMI in the same df
df_us = pd.merge(pmi_us, el_us, how='left', left_index=True, right_index=True)
df_us = pd.merge(df_us, brent, how='left', left_index=True, right_index=True)
df_us = pd.merge(df_us, wti, how='left', left_index=True, right_index=True)
df_us = df_us.dropna()



'''
Developing dynamic models (SARIMA with explanatory variables) 
'''
# Formally prove that only one differencing is needed
df_no = df_no.dropna()
ndiffs(df_no.pmi, test='adf')
nsdiffs(df_no.pmi, test='ch', m=12)

# Adding direction column in all data frames (1 if PMI goes up, 0 if down)
df_no['dir'] = [1 if x > 0 else 0 for x in df_no.pmi - df_no.pmi.shift(1)]
df_dk['dir'] = [1 if x > 0 else 0 for x in df_dk.pmi - df_dk.pmi.shift(1)]
df_uk['dir'] = [1 if x > 0 else 0 for x in df_uk.pmi - df_uk.pmi.shift(1)]
df_us['dir'] = [1 if x > 0 else 0 for x in df_us.pmi - df_us.pmi.shift(1)]


# Need to find ARIMA terms for all countries. Using exog with only previous periods (only lags)
n_test_obs = 24
# Norway
df_no_train = df_no.iloc[:-n_test_obs,:]
df_no_test = df_no.iloc[-n_test_obs:,:]
exog_no_train = df_no_train.drop(['dir', 'eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_no_test = df_no_test.drop(['dir', 'eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
#model_no = auto_arima(df_no_train.pmi, m = 12, exogenous=exog_no_train.to_numpy(), d=1,
#                             max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
#                             maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic')
#model_no.summary()
arima_no_without_diff = SARIMAX(df_no_train.pmi, order=(2,0,1), seasonal_order=(2,0,2,12), exog=exog_no_train.astype('float'))
arima_no = SARIMAX(df_no_train.pmi, order=(2,1,2), seasonal_order=(1,0,1,12), exog=exog_no_train.astype('float'))
res_no = arima_no.fit(maxiter=100, disp=0)
res_no_without_diff = arima_no_without_diff.fit(maxiter=100, disp=0)

# Denmark
df_dk_train = df_dk.iloc[:-n_test_obs,:]
df_dk_test = df_dk.iloc[-n_test_obs:,:]
exog_dk_train = df_dk_train.drop(['dir', 'eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_dk_test = df_dk_test.drop(['dir', 'eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
#model_dk = auto_arima(df_dk_train.pmi, m = 12, exogenous=exog_dk_train.to_numpy(), d=1,
#                             max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
#                             maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic')
#model_dk.summary()
arima_dk_without_diff = SARIMAX(df_dk_train.pmi, order=(1,0,1), seasonal_order=(0,0,1,12), exog=exog_dk_train.astype('float'))
arima_dk = SARIMAX(df_dk_train.pmi, order=(0,1,1), seasonal_order=(0,0,1,12), exog=exog_dk_train.astype('float'))
res_dk = arima_dk.fit(maxiter=100, disp=0)
res_dk_without_diff = arima_dk_without_diff.fit(maxiter=100, disp=0)

# UK
df_uk.index.freq='MS'
df_uk_train = df_uk.iloc[:-n_test_obs,:]
df_uk_test = df_uk.iloc[-n_test_obs:,:]
exog_uk_train = df_uk_train.drop(['dir', 'monthyear', 'gbp_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_uk_test = df_uk_test.drop(['dir', 'monthyear', 'gbp_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
#model_uk = auto_arima(df_uk_train.pmi, m = 12, exogenous=exog_uk_train.to_numpy(), d=1,
#                             max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
#                             maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic')
#model_uk.summary()
arima_uk_without_diff = SARIMAX(df_uk_train.pmi, order=(1,0,2), exog=exog_uk_train.astype('float'))
arima_uk = SARIMAX(df_uk_train.pmi, order=(2,1,2), exog=exog_uk_train.astype('float'))
res_uk = arima_uk.fit(maxiter=100, disp=0)
res_uk_without_diff = arima_uk_without_diff.fit(maxiter=100, disp=0)


# US
df_us.index.freq='MS'
df_us_train = df_us.iloc[:-n_test_obs,:]
df_us_test = df_us.iloc[-n_test_obs:,:]
exog_us_train = df_us_train.drop(['dir', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_us_test = df_us_test.drop(['dir', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
#model_us = auto_arima(df_us_train.pmi, m = 12, exogenous=exog_us_train.to_numpy(), d=1,
#                             max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
#                             maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic')
#model_us.summary()
arima_us_without_diff = SARIMAX(df_us_train.pmi, order=(1,0,0), seasonal_order=(0,0,1,12), exog=exog_us_train.astype('float'))
arima_us = SARIMAX(df_us_train.pmi, order=(0,1,0), seasonal_order=(0,0,1,12), exog=exog_us_train.astype('float'))
res_us = arima_us.fit(maxiter=100, disp=0)
res_us_without_diff = arima_us_without_diff.fit(maxiter=100, disp=0)

# Creating summary table for coefficients and p-values
res_dict = {'Norway':res_no, 'Denmark':res_dk, 'UK':res_uk, 'US':res_us}

names = []
coefs = []
cntry = []
for i in range(len(res_dict.values())):
    res = list(res_dict.values())[i]
    tab = res.summary().tables[1]
    country = list(res_dict.keys())[i]
    for row in tab[1:]:
        name = str(row[0])
        coef = float(str(row[1]))
        pval = float(str(row[4]))

        names.append(name)
        coefs.append('{} ({})'.format(coef, pval))
        cntry.append(country)


df_long = pd.DataFrame(data={'names':names, 'coefs':coefs, 'country':cntry})
df_unsorted = pd.pivot(df_long, values='coefs', columns='country', index='names')
df = df_unsorted.reindex(index=df_long['names']).iloc[:13,:]
df.index.name = ''
df.at['aic', 'Norway'] = res_no.info_criteria('aic')
df.at['aic', 'Denmark'] = res_dk.info_criteria('aic')
df.at['aic', 'UK'] = res_uk.info_criteria('aic')
df.at['aic', 'US'] = res_us.info_criteria('aic')

df.at['ARIMA order', 'Norway'] = '{}{}'.format(arima_no.order, arima_no.seasonal_order)
df.at['ARIMA order', 'Denmark'] = '{}{}'.format(arima_dk.order, arima_dk.seasonal_order)
df.at['ARIMA order', 'UK'] = '{}{}'.format(arima_uk.order, arima_uk.seasonal_order)
df.at['ARIMA order', 'US'] = '{}{}'.format(arima_us.order, arima_us.seasonal_order)
print(df.to_string())


# Calculating RMSE
rmse_no =(((res_no.get_forecast(n_test_obs, exog=exog_no_test.astype('float')).summary_frame()['mean']-df_no_test.pmi)**2).mean())**0.5
rmse_dk =(((res_dk.get_forecast(n_test_obs, exog=exog_dk_test.astype('float')).summary_frame()['mean']-df_dk_test.pmi)**2).mean())**0.5
rmse_uk =(((res_uk.get_forecast(n_test_obs, exog=exog_uk_test.astype('float')).summary_frame()['mean']-df_uk_test.pmi)**2).mean())**0.5
rmse_us =(((res_us.get_forecast(n_test_obs, exog=exog_us_test.astype('float')).summary_frame()['mean']-df_us_test.pmi)**2).mean())**0.5

rmse_no_wd =(((res_no_without_diff.get_forecast(n_test_obs, exog=exog_no_test.astype('float')).summary_frame()['mean']-df_no_test.pmi)**2).mean())**0.5
rmse_dk_wd =(((res_dk_without_diff.get_forecast(n_test_obs, exog=exog_dk_test.astype('float')).summary_frame()['mean']-df_dk_test.pmi)**2).mean())**0.5
rmse_uk_wd =(((res_uk_without_diff.get_forecast(n_test_obs, exog=exog_uk_test.astype('float')).summary_frame()['mean']-df_uk_test.pmi)**2).mean())**0.5
rmse_us_wd =(((res_us_without_diff.get_forecast(n_test_obs, exog=exog_us_test.astype('float')).summary_frame()['mean']-df_us_test.pmi)**2).mean())**0.5

rmse_no_naive = ((([df_no_train.pmi[-1]]*n_test_obs-df_no_test.pmi)**2).mean())**0.5
rmse_dk_naive = ((([df_dk_train.pmi[-1]]*n_test_obs-df_dk_test.pmi)**2).mean())**0.5
rmse_uk_naive = ((([df_uk_train.pmi[-1]]*n_test_obs-df_uk_test.pmi)**2).mean())**0.5
rmse_us_naive = ((([df_us_train.pmi[-1]]*n_test_obs-df_us_test.pmi)**2).mean())**0.5

rmse = [rmse_no, rmse_dk, rmse_uk, rmse_us]
rmse_wd = [rmse_no_wd, rmse_dk_wd, rmse_uk_wd, rmse_us_wd]
rmse_naive = [rmse_no_naive, rmse_dk_naive, rmse_uk_naive, rmse_us_naive]

rmse_df = pd.DataFrame(data={'d=1':rmse, 'd=0':rmse_wd, 'naive':rmse_naive})
rmse_df.index = ['Norway', 'Denmark', 'UK', 'US']

def rmse_column(df, n_test_obs, order, seasonal_order):

    rmse_list = []
    top = []
    bottom = []
    for i in range(1, n_test_obs+1):
        train = df.iloc[:-i,:]
        test = df.iloc[-i:,:]
        exog_train = train[['el_lag_1', 'el_lag_2', 'brent_lag_1', 'brent_lag_2', 'wti_lag_1', 'wti_lag_2']]
        exog_test = test[['el_lag_1', 'el_lag_2', 'brent_lag_1', 'brent_lag_2', 'wti_lag_1', 'wti_lag_2']]

        arima = SARIMAX(train.pmi, order=order, seasonal_order=seasonal_order, exog=exog_train.astype('float'))
        res = arima.fit(maxiter=100, disp=0)

        f = res.get_forecast(i, exog=exog_test.astype('float')).summary_frame()
        resid = abs(f['mean']-test.pmi)
        rmse = ((resid**2).mean())**0.5

        rmse_list.append(rmse)
        top.append(max(resid))
        bottom.append(min(resid))


        print('Forecasting {} observations'.format(i))
    final_df = pd.DataFrame(data={'rmse':rmse_list, 'top':top, 'bottom':bottom})
    return final_df

rmse_no = rmse_column(df_no, 24, arima_no.order, arima_no.seasonal_order)
rmse_dk = rmse_column(df_dk, 24, arima_dk.order, arima_dk.seasonal_order)
rmse_uk = rmse_column(df_uk, 24, arima_uk.order, arima_uk.seasonal_order)
rmse_us = rmse_column(df_us, 24, arima_us.order, arima_us.seasonal_order)

rmse_no_wd = rmse_column(df_no, 24, arima_no_without_diff.order, arima_no_without_diff.seasonal_order)
rmse_dk_wd = rmse_column(df_dk, 24, arima_dk_without_diff.order, arima_dk_without_diff.seasonal_order)
rmse_uk_wd = rmse_column(df_uk, 24, arima_uk_without_diff.order, arima_uk_without_diff.seasonal_order)
rmse_us_wd = rmse_column(df_us, 24, arima_us_without_diff.order, arima_us_without_diff.seasonal_order)

# Plotting RMSE
def plot_rmse(rmse, rmse_wd, title, filename, ax):
    col1 = 'black'
    col2 = 'gray'
    ax.errorbar(x=rmse.index, y=rmse.rmse, yerr=[rmse.rmse-rmse.bottom, rmse.top-rmse.rmse],
                fmt='-o', capsize=5, errorevery=3, color=col1, label='d=1')
    ax.errorbar(x=rmse_wd.index, y=rmse_wd.rmse, yerr=[rmse_wd.rmse-rmse_wd.bottom, rmse_wd.top-rmse_wd.rmse],
                fmt='--o', capsize=5, errorevery=3, color=col2, fillstyle='none', label='d=0')
    handles, labels = ax.get_legend_handles_labels()
    full_line = mlines.Line2D([], [], linestyle='-', color=col1, label='d = 1')
    dot_line = mlines.Line2D([], [], linestyle='--', color=col2, label='d = 0')
    ax.legend(handles=[full_line, dot_line], loc='upper left')
    ax.set_title(title, fontsize=18)
    ax.set_ylabel('RMSE')

fig, axes = plt.subplots(nrows=4, figsize=(8,10))
plot_rmse(rmse_no, rmse_no_wd, 'RMSE - Norway', 'rmse_no', axes[0])
plot_rmse(rmse_dk, rmse_dk_wd, 'RMSE - Denmark', 'rmse_dk', axes[1])
plot_rmse(rmse_uk, rmse_uk_wd, 'RMSE - UK', 'rmse_uk', axes[2])
plot_rmse(rmse_us, rmse_us_wd, 'RMSE - US', 'rmse_us', axes[3])
plt.savefig('plots/rmse.png')
plt.show()

# Plotting residuals
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10,6))
plt.subplots_adjust(hspace=0.8)
ax[0,0].plot(res_no.resid[1:])
ax[0,1].hist(res_no.resid[1:])

ax[1,0].plot(res_dk.resid[1:])
ax[1,1].hist(res_dk.resid[1:])

ax[2,0].plot(res_uk.resid[1:])
ax[2,1].hist(res_uk.resid[1:])

ax[3,0].plot(res_us.resid[1:])
ax[3,1].hist(res_us.resid[1:])

ax[0,0].set_title('Norway')
ax[1,0].set_title('Denmark')
ax[2,0].set_title('UK')
ax[3,0].set_title('US')

ax[0,0].tick_params(axis='x', labelrotation=0)
ax[1,0].tick_params(axis='x', labelrotation=0)
ax[2,0].tick_params(axis='x', labelrotation=0)
ax[3,0].tick_params(axis='x', labelrotation=0)
plt.suptitle('Residuals', fontweight='bold')
plt.savefig('plots/residuals.png')
plt.show()

# Plotting residuals of without differencing
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(8,8))
ax[0,0].plot(res_no_without_diff.resid[1:])
ax[0,1].hist(res_no_without_diff.resid[1:])

ax[1,0].plot(res_dk_without_diff.resid[1:])
ax[1,1].hist(res_dk_without_diff.resid[1:])

ax[2,0].plot(res_uk_without_diff.resid[1:])
ax[2,1].hist(res_uk_without_diff.resid[1:])

ax[3,0].plot(res_us_without_diff.resid[1:])
ax[3,1].hist(res_us_without_diff.resid[1:])

ax[0,0].set_title('Norway')
ax[1,0].set_title('Denmark')
ax[2,0].set_title('UK')
ax[3,0].set_title('US')

ax[0,0].tick_params(axis='x', labelrotation=25)
ax[1,0].tick_params(axis='x', labelrotation=25)
ax[2,0].tick_params(axis='x', labelrotation=25)
ax[3,0].tick_params(axis='x', labelrotation=25)
plt.suptitle('Residuals (d=0)', fontweight='bold')
plt.savefig('plots/residuals_wd.png')
plt.show()

# Plotting forecasts
fig, ax = plt.subplots(figsize=(12,12), ncols=1, nrows=4)
plt.subplots_adjust(hspace=0.4)
plot_multiple_forecast(res_no, res_no_without_diff, df_no_train.pmi, df_no_test.pmi, exog_no_test, 'Norway', ax[0])
plot_multiple_forecast(res_dk, res_dk_without_diff, df_dk_train.pmi, df_dk_test.pmi, exog_dk_test, 'Denmark', ax[1])
plot_multiple_forecast(res_uk, res_uk_without_diff, df_uk_train.pmi, df_uk_test.pmi, exog_uk_test, 'UK', ax[2])
plot_multiple_forecast(res_us, res_us_without_diff, df_us_train.pmi, df_us_test.pmi, exog_us_test, 'US', ax[3])
plt.suptitle('Forecasts', fontweight='bold', fontsize=18)
handles, labels = ax[0].get_legend_handles_labels()
lg = ax[3].legend(handles, labels, ncol=5, loc='lower center', bbox_to_anchor=(0.5,-0.4))
plt.savefig('plots/forecasts.png', bbox_extra_artists =(lg,), bbox_inches='tight')
plt.show()
'''
Developing Random Forest model to predict if PMI will go up or down based on lagged oil price and electricity variables
'''
# Creating new test sets
n_test_obs = 48

# Norway
df_no_train = df_no.iloc[:-n_test_obs,:]
df_no_test = df_no.iloc[-n_test_obs:,:]
exog_no_train = df_no_train.drop(['dir', 'eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_no_test = df_no_test.drop(['dir', 'eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)

# Denmark
df_dk_train = df_dk.iloc[:-n_test_obs,:]
df_dk_test = df_dk.iloc[-n_test_obs:,:]
exog_dk_train = df_dk_train.drop(['dir', 'eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_dk_test = df_dk_test.drop(['dir', 'eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)

# UK
df_uk_train = df_uk.iloc[:-n_test_obs,:]
df_uk_test = df_uk.iloc[-n_test_obs:,:]
exog_uk_train = df_uk_train.drop(['dir', 'monthyear', 'gbp_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_uk_test = df_uk_test.drop(['dir', 'monthyear', 'gbp_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)

# US
df_us_train = df_us.iloc[:-n_test_obs,:]
df_us_test = df_us.iloc[-n_test_obs:,:]
exog_us_train = df_us_train.drop(['dir', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_us_test = df_us_test.drop(['dir', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)

# Plot random forest performance with varying m
def accuracy_column(df, n_test_obs):
    accuracy_list = []
    for i in range(6):
        train = df.iloc[:-n_test_obs,:]
        test = df.iloc[-n_test_obs:,:]
        exog_train = train[['el_lag_1', 'el_lag_2', 'brent_lag_1', 'brent_lag_2', 'wti_lag_1', 'wti_lag_2']]
        exog_test = test[['el_lag_1', 'el_lag_2', 'brent_lag_1', 'brent_lag_2', 'wti_lag_1', 'wti_lag_2']]

        rf = RandomForestClassifier(n_estimators=128, bootstrap=True, max_features=i+1, random_state=86)
        rf.fit(X=exog_train, y=train.dir)
        preds = rf.predict(exog_test)

        accuracy = accuracy_score(y_true=test.dir, y_pred=preds)
        accuracy_list.append(accuracy)
    return accuracy_list

# Model Accuracy, how often is the classifier correct?
acc_no = accuracy_column(df_no, 24)
acc_dk = accuracy_column(df_dk, 24)
acc_uk = accuracy_column(df_uk, 24)
acc_us = accuracy_column(df_us, 24)

fig, ax = plt.subplots()
x_axis = range(1, 7)
ax.plot(x_axis, acc_no, '-.o', label='Norway')
ax.plot(x_axis, acc_dk, '-.o', label='Denmark')
ax.plot(x_axis, acc_uk, '-.o', label='UK')
ax.plot(x_axis, acc_us, '-.o', label='US')
plt.axvline(x=math.sqrt(6), linestyle='--', color='gray', linewidth=1)
ax.set_ylabel('Out-of-sample accuracy')
ax.set_xlabel('Predictor subset')
ax.set_title('Out-of-sample accuracy with differing m', fontdict={'size':18})
plt.legend(loc='best')
plt.savefig('plots/oos_accuracy.png')
plt.show()

### Norway
rf_no = RandomForestClassifier(n_estimators=128, bootstrap=True, max_features='auto', random_state=86)
rf_no.fit(X=exog_no_train, y=df_no_train.dir)

preds_no = rf_no.predict(exog_no_test)
rf_no.predict_proba(exog_no_test) # probabilities calculated by the Random Forest model
conf_mat_no = confusion_matrix(y_true=df_no_test.dir, y_pred=preds_no, labels=[0,1])


### Denmark
rf_dk = RandomForestClassifier(n_estimators=128, bootstrap=True, max_features='auto', random_state=86)
rf_dk.fit(X=exog_dk_train, y=df_dk_train.dir)

preds_dk = rf_dk.predict(exog_dk_test)
rf_dk.predict_proba(exog_dk_test) # probabilities calculated by the Random Forest model
conf_mat_dk = confusion_matrix(y_true=df_dk_test.dir, y_pred=preds_dk, labels=[0,1])

### UK
rf_uk = RandomForestClassifier(n_estimators=128, bootstrap=True, max_features='auto', random_state=86)
rf_uk.fit(X=exog_uk_train, y=df_uk_train.dir)

preds_uk = rf_uk.predict(exog_uk_test)
rf_uk.predict_proba(exog_uk_test) # probabilities calculated by the Random Forest model
conf_mat_uk = confusion_matrix(y_true=df_uk_test.dir, y_pred=preds_uk, labels=[0,1])

### US
rf_us = RandomForestClassifier(n_estimators=128, bootstrap=True, max_features='auto', random_state=86)
rf_us.fit(X=exog_us_train, y=df_us_train.dir)

preds_us = rf_us.predict(exog_us_test)
rf_us.predict_proba(exog_us_test) # probabilities calculated by the Random Forest model
conf_mat_us = confusion_matrix(y_true=df_us_test.dir, y_pred=preds_us, labels=[0,1])

# Plotting confusion matrices
fix, ax = plt.subplots(2,2)
plt.subplots_adjust(hspace=0.6)
plot_confusion_matrix(rf_no, X=exog_no_test, y_true=df_no_test.dir, ax=ax[0,0])
plot_confusion_matrix(rf_dk, X=exog_dk_test, y_true=df_dk_test.dir, ax=ax[0,1])
plot_confusion_matrix(rf_uk, X=exog_uk_test, y_true=df_uk_test.dir, ax=ax[1,0])
plot_confusion_matrix(rf_us, X=exog_us_test, y_true=df_us_test.dir, ax=ax[1,1])
ax[0,0].set_title('Norway')
ax[0,1].set_title('Denmark')
ax[1,0].set_title('UK')
ax[1,1].set_title('US')
plt.savefig('plots/conf_mat.png')
plt.show()

## Feature importance
# Norway
fi_no = pd.DataFrame({'feature': list(exog_no_test.columns),
                   'importance': rf_no.feature_importances_}). \
    sort_values('importance', ascending = False)

# Denmark
fi_dk = pd.DataFrame({'feature': list(exog_dk_test.columns),
                   'importance': rf_dk.feature_importances_}). \
    sort_values('importance', ascending = False)

# UK
fi_uk = pd.DataFrame({'feature': list(exog_uk_test.columns),
                   'importance': rf_uk.feature_importances_}). \
    sort_values('importance', ascending = False)

# US
fi_us = pd.DataFrame({'feature': list(exog_us_test.columns),
                   'importance': rf_us.feature_importances_}). \
    sort_values('importance', ascending = False)

# Plotting feature importance (in four plots)
fig, ax = plt.subplots(2,2)
plt.subplots_adjust(hspace=0.5)
ax[0,0].barh(fi_no.feature, fi_no.importance, label='Norway')
ax[0,1].barh(fi_dk.feature, fi_dk.importance, label='Denmark')
ax[1,0].barh(fi_uk.feature, fi_uk.importance, label='UK')
ax[1,1].barh(fi_us.feature, fi_us.importance, label='US')
plt.legend(loc='best')
plt.show()

# Plotting feature importance (in one plot)
fi_no.sort_values(by='feature', inplace=True)
fi_dk.sort_values(by='feature', inplace=True)
fi_uk.sort_values(by='feature', inplace=True)
fi_us.sort_values(by='feature', inplace=True)
fi_all = pd.DataFrame(data={'no': fi_no.importance, 'dk': fi_dk.importance,
                            'uk': fi_uk.importance, 'us': fi_us.importance})
fi_all.index=fi_no.feature

ax=fi_all.plot.barh()
ax.set_ylabel('Feature')
ax.set_xlabel('Feature importance')
ax.set_title('Feature importance', fontdict={'size':18})
plt.tight_layout()
plt.savefig('plots/feat_imp.png', )
plt.show()