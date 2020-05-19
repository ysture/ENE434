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

# Creating lagged series
brent['lag_1'] = brent['usd_per_barrel'].shift(-1)
brent['lag_2'] = brent['usd_per_barrel'].shift(-2)
wti['lag_1'] = wti['usd_per_barrel'].shift(-1)
wti['lag_2'] = wti['usd_per_barrel'].shift(-2)

el_no['lag_1'] = el_no['usd_per_MWh'].shift(-1)
el_no['lag_2'] = el_no['usd_per_MWh'].shift(-2)
el_dk['lag_1'] = el_dk['usd_per_MWh'].shift(-1)
el_dk['lag_2'] = el_dk['usd_per_MWh'].shift(-2)
el_uk['lag_1'] = el_uk['usd_per_MWh'].shift(-1)
el_uk['lag_2'] = el_uk['usd_per_MWh'].shift(-2)
el_us['lag_1'] = el_us['usd_per_MWh'].shift(-1)
el_us['lag_2'] = el_us['usd_per_MWh'].shift(-2)
el_ge['lag_1'] = el_ge['usd_per_MWh'].shift(-1)
el_ge['lag_2'] = el_ge['usd_per_MWh'].shift(-2)


# Making data for all countries the same length. The length of the time series
# is decided by the shortest time series length for each country
    # Norway
el_no.shape
pmi_no.shape
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
Developing dynamic models (SARIMA with explanatory variables) 
'''


model_auto = auto_arima(cons_dk, m = 7,
                        max_order = None, max_p = 5, max_q = 5, max_d = 1, max_P = 3, max_Q = 5, max_D = 2,
                        maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'aic',
                        out_of_sample = int(len(cons_dk*0.2)))