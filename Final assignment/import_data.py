'''File to import data used in final ENE434 Assignment
Mangler: Gasspriser
'''
import pandas as pd
import numpy as np
from tabula import read_pdf
import os
import matplotlib.pyplot as plt
from datetime import datetime, date
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima

#TODO Finn bedre kilde til strømpriser fra Tyskland (sjekk link fra Hendrik)

# Importing currency exchange rates
cur = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/macrobond_currency.csv',
                  header=0,
                  names=['month', 'eur_per_usd', 'gbp_per_usd'])
cur['month'] = pd.to_datetime(cur['month'])
currency_dict = {'eur':'eur_per_usd', 'gbp':'gbp_per_usd'}

# Helper functions
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

# Creating function to plot forecasts
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


### PMI
# Germany
pmi_ge = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/germany.markit-manufacturing-pmi.csv',
                      sep='\t',
                      usecols=[0,1],
                      header=0,
                      names=['month', 'pmi'])
pmi_ge['month'] = pd.to_datetime(pmi_ge['month'], dayfirst=True)
pmi_ge = remove_projections(pmi_ge)

# Shifting month one month back
pmi_ge['month'] = shift_month_back(pmi_ge)


## Norway
# Ordinary
pmi_no_list = read_pdf(
    'C:\\Users\\Yngve\\Google Drive\\Skolerelatert\\NHH\\Master\\ENE434\\Final assignment\\input\\PMI_norway_pdf.pdf',
    pages=[1, 2, 3, 4, 5], multiple_tables=False,
    pandas_options={'header': None,
                    'usecols':[0,1],
                    'names': ['month', 'pmi']})

# Seasonally adjusted
pmi_no_seasadj_list = read_pdf(
    'C:\\Users\\Yngve\\Google Drive\\Skolerelatert\\NHH\\Master\\ENE434\\Final assignment\\input\\PMI_norway_pdf.pdf',
    pages=[6, 7, 8, 9, 10], multiple_tables=False,
    pandas_options={'header': None,
                    'usecols':[0,1],
                    'names': ['month', 'pmi']})


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

## NordPool (Norway, Netherlands and Germany)
nordpool_files = [f for f in os.listdir('input/') if f.startswith('elspot-prices_')]
def import_nordpool():
    df = pd.DataFrame()
    for f in nordpool_files:
        dfTemp = pd.read_csv('input/{}'.format(f), engine='python', header=2)
        df = pd.concat([df, dfTemp], axis=0)
    df = df.reset_index(drop=True).rename(columns={'Unnamed: 0':'month'})
    df = df.stack().str.replace(',','.').unstack()
    return df
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

# Germany (only 9 months). Only available nine months for Netherlands as well
el_ge = np_df[['DE-LU']].astype('float')
el_ge = pd.concat([np_df.month, el_ge], axis=1)
el_ge.columns = ['month', 'eur_per_MWh']
el_ge.dropna(inplace=True)

# US
us_filenames = os.listdir('input/ice_electric-historical')


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
el_ge = convert_currency(el_ge)

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
pmi_ge = convert_dtypes(pmi_ge)
pmi_uk = convert_dtypes(pmi_uk)
pmi_us = convert_dtypes(pmi_us)


'''
Descriptive statistics in this order:
1. PMI
2. Electricity prices
3. Oil prices
'''

### PMI
# In a 2x3 grid (all data available)
pmi_dk_decomp = seasonal_decompose(pmi_dk.pmi, period=12).trend.dropna()
pmi_dk_seasadj = pd.DataFrame({'month':pmi_dk.month[pmi_dk.index.isin(pmi_dk_decomp.index)],
                              'pmi':pmi_dk_decomp})


pmi_dict = {'Norway':pmi_no,
            'Norway (seas. adj)':pmi_no_seasadj,
            'Denmark':pmi_dk,
            'Germany':pmi_ge,
            'UK':pmi_uk,
            'US':pmi_us}


fig, ax = plt.subplots(ncols=2, nrows=3)
i=0
for row in ax:
    for col in row:
        df = list(pmi_dict.values())[i]
        col.plot(df['month'], df['pmi'])
        title = list(pmi_dict.keys())[i] + ' PMI'
        col.set_title(title)
        col.tick_params(axis='x', labelrotation=45)
        # Next iteration
        i+=1
plt.show()

# In a single plot (from 2008)
start_date = np.datetime64(date(2008, 1, 1))
fig, ax = plt.subplots()
i=0
for i in range(len(pmi_dict.values())):
    df = list(pmi_dict.values())[i]
    df = df[df['month']> start_date]
    label = list(pmi_dict.keys())[i]
    ax.plot(df['month'], df['pmi'], label=label)
    ax.set_title('PMI 2008-2020', fontdict={'size':18})
plt.legend(loc='best')
plt.show()

### Electricity prices
el_no = el_no[['month', 'usd_per_MWh', 'eur_per_MWh']]
el_uk = el_uk[['month', 'usd_per_MWh', 'gbp_per_MWh']]
el_us = el_us[['month', 'usd_per_MWh']]
el_dk = el_dk[['month', 'usd_per_MWh', 'eur_per_MWh']]

el_dict = {'Norway':el_no,
           'UK':el_uk,
           'US':el_us,
           'Denmark':el_dk}
# In a single plot
start_date = np.datetime64(date(2005, 1, 1))
fig, ax = plt.subplots()
i=0
for i in range(len(el_dict.values())):
    df = list(el_dict.values())[i]
    df = df[df['month']> start_date]
    df.dropna(inplace=True)
    label = list(el_dict.keys())[i]
    ax.plot(df['month'], df['usd_per_MWh'], label=label)
    ax.set_title('$/MWh 2005-2020', fontdict={'size':18})
plt.legend(loc='best')
plt.show()


### Oil prices
fig, ax = plt.subplots()
ax.plot(brent.month, brent.usd_per_barrel, label='Brent')
ax.plot(wti.month, wti.usd_per_barrel, label='WTI')
ax.set_title('$/barrel 1987-2020')
plt.legend(loc='best')
plt.show()


'''
Investigating seasonality for all time series, should any of the series be seasonally adjusted?
1. PMI
2. Electricity
3. Oil prices
'''
# Helper function to plot decomposition and the different decomposed elements
def plot_decomposition(df, column_index, plot_title):
    try:
        df = df.dropna()
        value_column = df.iloc[:, column_index]
        decomp = seasonal_decompose(value_column, period=12)
        #TODO prøv med period=52 siden man har ukentlig data

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

# PMI
plot_decomposition(pmi_dk, 1, 'Denmark PMI')
plot_decomposition(pmi_no, 1, 'Norway PMI')
plot_decomposition(pmi_us, 1, 'US PMI')
plot_decomposition(pmi_uk, 1, 'UK PMI')
plot_decomposition(pmi_ge, 1, 'Germany PMI')

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
def plot_autocorrelation(dict, column_index):
    print('ADFuller test to check for stationarity (H0 is that there is non-stationarity):')
    for i in range(len(list(dict.values()))):
        df = list(dict.values())[i].dropna()
        title = list(dict.keys())[i]

        plot_pacf(df.iloc[:,column_index], ax=axes[i,0], title=title)
        plot_acf(df.iloc[:,column_index], ax=axes[i,1], title=title)

        # Print ADFuller test
        p_val = adfuller(df.iloc[:, column_index])[1]
        print('P-value of {c}: {p}'.format(c=title, p=p_val))

    fig.align_ylabels()
    plt.show()

# PMI
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(16,32))
plt.subplots_adjust(hspace=0.4)
title_size = 15
title_type = 'bold'
ylabel_size = 12

plot_autocorrelation(pmi_dict, 1)

# Electricity
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16,32))
plt.subplots_adjust(hspace=0.4)
title_size = 15
title_type = 'bold'
ylabel_size = 12
plot_autocorrelation(el_dict, 1)

# Oil
oil_dict = {'Brent':brent,
            'WTI': wti}

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,32))
plt.subplots_adjust(hspace=0.4)
title_size = 15
title_type = 'bold'
ylabel_size = 12
plot_autocorrelation(oil_dict, 1)

'''
Preparing data for modelling
'''

# Preparing data for modelling
# Creating lagged series
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
el_ge['el_lag_1'] = el_ge['usd_per_MWh'].shift(-1)
el_ge['el_lag_2'] = el_ge['usd_per_MWh'].shift(-2)

# Making data for all countries the same length. The length of the time series
# is decided by the shortest time series length for each country
pmi_no = pmi_no[pmi_no.month.isin(el_no.month)]
pmi_dk = pmi_dk[pmi_dk.month.isin(el_dk.month)]
pmi_uk = pmi_uk[pmi_uk.month.isin(el_uk.month)]
pmi_us = pmi_us[pmi_us.month.isin(el_us.month)]

# Making date index for all dataframes
def month_as_index(df):
    df.index = df.month
    df.drop(['month'], inplace=True, axis=1)
    return df

pmi_no = month_as_index(pmi_no)
pmi_dk = month_as_index(pmi_dk)
pmi_us = month_as_index(pmi_us)
pmi_uk = month_as_index(pmi_uk)

el_dk = month_as_index(el_dk)
el_no = month_as_index(el_no)
el_uk = month_as_index(el_uk)
el_us = month_as_index(el_us)
el_ge = month_as_index(el_ge)

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

# Creating one dataframe for each country to include exogenous variables and PMI in the same df
df_us = pd.merge(pmi_us, el_us, how='left', left_index=True, right_index=True)
df_us = pd.merge(df_us, brent, how='left', left_index=True, right_index=True)
df_us = pd.merge(df_us, wti, how='left', left_index=True, right_index=True)
df_us = df_us.dropna()

# Creating one dataframe for each country to include exogenous variables and PMI in the same df
df_ge = pd.merge(pmi_ge, el_ge, how='left', left_index=True, right_index=True)
df_ge = pd.merge(df_ge, brent, how='left', left_index=True, right_index=True)
df_ge = pd.merge(df_ge, wti, how='left', left_index=True, right_index=True)
df_ge = df_ge.dropna()



'''
Developing dynamic models (SARIMA with explanatory variables) 
'''
# Norway
model_auto = auto_arima(pmi_no.pmi, m = 12,
                        max_order = None, max_p = 5, max_q = 5, max_d = 1, max_P = 3, max_Q = 5, max_D = 2,
                        maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic',
                        out_of_sample = int(pmi_no.shape[0]*0.2))
model_auto.summary()
arima_auto = SARIMAX(pmi_no, order=(3,0,3))
results_auto = arima_auto.fit(maxiter=100, disp=0)
forecast_plot(pmi_no, arima_auto, forecasts=12, y_label="PMI")

# Norway with exogeneous variables
from pmdarima.arima.utils import ndiffs
from pmdarima.arima.utils import nsdiffs

# Formally prove that only one differencing is needed
df_no = df_no.dropna()
ndiffs(df_no.pmi, test='adf')
nsdiffs(df_no.pmi, test='ch', m=12)

'''
Preparing dynamic model (trying out with exogenous variables from both current and previous periods)
# Dynamic model with exogenous variables (including current period)
exog = df_no.drop(['eur_per_MWh', 'pmi'], axis=1)
model_exog = auto_arima(df_no.pmi, m = 12, exogenous=exog.to_numpy(),
                        max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
                        maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic',
                        out_of_sample = int(df_no.shape[0]*0.2))
model_exog.summary()
arima_exog = SARIMAX(df_no.pmi, order=(1,0,2), seasonal_order=(3,0,3,12), exog=exog.astype('float'))
results_exog = arima_exog.fit(maxiter=300)
forecasts=12
f = results_exog.get_forecast(forecasts, exog=exog.iloc[-forecasts:,:].astype('float')).summary_frame()
f_ix = f.index
pred_mean = f['mean'] # prediction mean
forecast_plot(df_no.pmi, arima_auto, forecasts=30, y_label="PMI", exog_test=exog) # plotting forecast

# Dynamic model with only previous periods for exogenous variables
exog_prev = df_no.drop(['eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
model_exog_prev = auto_arima(df_no.pmi, m = 12, exogenous=exog_prev.to_numpy(),
                             max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
                             maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic',
                             out_of_sample = int(df_no.shape[0]*0.2))
model_exog_prev.summary()
arima_exog_prev = SARIMAX(df_no.pmi, order=(1,0,1), seasonal_order=(3,0,3,12), exog=exog.astype('float'))
results_exog_prev = arima_exog_prev.fit(maxiter=300)
'''


# Need to find ARIMA terms for all countries. Using exog with only previous periods (only lags)
n_test_obs = 12
# Norway
df_no.index = pd.DatetimeIndex(df_no.index).to_period('M')
df_no_train = df_no.iloc[:-n_test_obs,:]
df_no_test = df_no.iloc[-n_test_obs:,:]
exog_no_train = df_no_train.drop(['eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_no_test = df_no_test.drop(['eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
#model_no = auto_arima(df_no_train.pmi, m = 12, exogenous=exog_no_train.to_numpy(),
#                             max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
#                             maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic')
#model_no.summary()
arima_no = SARIMAX(df_no_train.pmi, order=(2,0,1), seasonal_order=(2,0,2,12), exog=exog_no_train.astype('float'))
res_no = arima_no.fit(maxiter=100, disp=0)
f = res_no.get_forecast(n_test_obs, exog=exog_no_test.iloc[-n_test_obs:,:].astype('float')).summary_frame()

# Denmark
df_dk.index = pd.DatetimeIndex(df_dk.index).to_period('M')
df_dk_train = df_dk.iloc[:-n_test_obs,:]
df_dk_test = df_dk.iloc[-n_test_obs:,:]
exog_dk_train = df_dk_train.drop(['eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_dk_test = df_dk_test.drop(['eur_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
#model_dk = auto_arima(df_dk_train.pmi, m = 12, exogenous=exog_dk_train.to_numpy(),
#                             max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
#                             maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic')
#model_dk.summary()
arima_dk = SARIMAX(df_dk_train.pmi, order=(1,0,1), seasonal_order=(0,0,1,12), exog=exog_dk_train.astype('float'))
res_dk = arima_dk.fit(maxiter=100, disp=0)
f = res_dk.get_forecast(n_test_obs, exog=exog_dk_test.iloc[-n_test_obs:,:].astype('float')).summary_frame()

# UK
df_uk.index = pd.DatetimeIndex(df_uk.index).to_period('M')
df_uk_train = df_uk.iloc[:-n_test_obs,:]
df_uk_test = df_uk.iloc[-n_test_obs:,:]
exog_uk_train = df_uk_train.drop(['monthyear', 'gbp_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_uk_test = df_uk_test.drop(['monthyear', 'gbp_per_MWh', 'pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
#model_uk = auto_arima(df_uk_train.pmi, m = 12, exogenous=exog_uk_train.to_numpy(),
#                             max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
#                             maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic')
#model_uk.summary()
arima_uk = SARIMAX(df_uk_train.pmi, order=(1,0,2), exog=exog_uk_train.astype('float'))
res_uk = arima_uk.fit(maxiter=100, disp=0)
f = res_uk.get_forecast(n_test_obs, exog=exog_uk_test.iloc[-n_test_obs:,:].astype('float')).summary_frame()
f.index = df_uk_test.index


# US
df_us.index = pd.DatetimeIndex(df_us.index).to_period('M')
df_us_train = df_us.iloc[:-n_test_obs,:]
df_us_test = df_us.iloc[-n_test_obs:,:]
exog_us_train = df_us_train.drop(['pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
exog_us_test = df_us_test.drop(['pmi', 'usd_per_MWh', 'usd_per_barrel_x', 'usd_per_barrel_y'], axis=1)
#model_us = auto_arima(df_us_train.pmi, m = 12, exogenous=exog_us_train.to_numpy(),
#                             max_order = None, max_p = 5, max_q = 5, max_d = 4, max_P = 3, max_Q = 5, max_D = 4,
#                             maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic')
#model_us.summary()
arima_us = SARIMAX(df_us_train.pmi, order=(1,0,0), seasonal_order=(0,0,1,12), exog=exog_us_train.astype('float'))
res_us = arima_us.fit(maxiter=100, disp=0)
f = res_us.get_forecast(n_test_obs, exog=exog_us_test.iloc[-n_test_obs:,:].astype('float')).summary_frame()
f.index = df_us_test.index


# Plotting forecasts
forecast_plot(df_no.pmi, arima_no, forecasts=n_test_obs, y_label="PMI", exog_test=exog_no_test)
forecast_plot(df_dk.pmi, arima_dk, forecasts=n_test_obs, y_label="PMI", exog_test=exog_dk_test)
forecast_plot(df_uk.pmi, arima_uk, forecasts=n_test_obs, y_label="PMI", exog_test=exog_uk_test)
forecast_plot(df_us.pmi, arima_us, forecasts=n_test_obs, y_label="PMI", exog_test=exog_us_test)

# Calculating RMSE
rmse_no =(((f['mean']-df_no_test.pmi)**2).mean())**0.5
rmse_dk =(((f['mean']-df_dk_test.pmi)**2).mean())**0.5
rmse_uk =(((f['mean']-df_uk_test.pmi)**2).mean())**0.5
rmse_us =(((f['mean']-df_us_test.pmi)**2).mean())**0.5



# TODO Lag et test set for å predikere 2019.
# TODO Lag dynamic models for de andre landene
# TODO Etter at alt annet (inkludert andre ML-modeller) er ferdig: Predikér resten av 2020 (om det er plass)

'''
Developing Random Forest model to predict if PMI will go up or down based on lagged oil price and electricity variables
'''
