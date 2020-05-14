'''File to import data used in final ENE434 Assignment
Mangler: Alle PMI-tall samt gasspriser.
'''
import pandas as pd
import numpy as np
from tabula import read_pdf
import os
import matplotlib.pyplot as plt
from datetime import datetime, date

#TODO Shift måned for tysk PMI én måned bakover
#TODO Finn bedre kilde til strømpriser fra Tyskland (sjekk link fra Hendrik)

### PMI
def remove_projections(df):
    df['monthyear'] = df.month.dt.strftime('%m-%Y')
    for x in df.monthyear.unique():
        if sum(df.monthyear == x) > 1:
            df.drop(df.month[df. month == max(df.month[df.monthyear == x])].index, axis=0, inplace=True)
    return df
# Germany
pmi_ger = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/germany.markit-manufacturing-pmi.csv',
                      sep='\t',
                      usecols=[0,1],
                      header=0,
                      names=['month', 'pmi'])
pmi_ger['month'] = pd.to_datetime(pmi_ger['month'], dayfirst=True)
pmi_ger = remove_projections(pmi_ger)

## Norway
# Ordinary
pmi_nor_list = read_pdf(
    'C:\\Users\\Yngve\\Google Drive\\Skolerelatert\\NHH\\Master\\ENE434\\Final assignment\\input\\PMI_norway_pdf.pdf',
    pages=[1, 2, 3, 4, 5], multiple_tables=False,
    pandas_options={'header': None,
                    'usecols':[0,1],
                    'names': ['month', 'pmi']})

# Seasonally adjusted
pmi_nor_seasadj_list = read_pdf(
    'C:\\Users\\Yngve\\Google Drive\\Skolerelatert\\NHH\\Master\\ENE434\\Final assignment\\input\\PMI_norway_pdf.pdf',
    pages=[6, 7, 8, 9, 10], multiple_tables=False,
    pandas_options={'header': None,
                    'usecols':[0,1],
                    'names': ['month', 'pmi']})


pmi_nor = pmi_nor_list[0].dropna()
pmi_nor['month'] = pd.to_datetime(pmi_nor['month'], dayfirst=True)
pmi_nor_seasadj = pmi_nor_seasadj_list[0].dropna()
pmi_nor_seasadj['month'] = pd.to_datetime(pmi_nor_seasadj['month'], dayfirst=True)

# UK
pmi_uk = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/united-kingdom.markit-manufacturing-pmi.csv',
                     sep='\t',
                     usecols=[0,1],
                     header=0,
                     names=['month', 'pmi'])
pmi_uk['month'] = pd.to_datetime(pmi_uk['month'], dayfirst=True)
pmi_uk = remove_projections(pmi_uk)

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
el_nor = np_df[['Oslo', 'Kr.sand', 'Bergen', 'Molde', 'Tr.heim', 'Tromsø']].astype('float')
el_nor = pd.concat([np_df.month, el_nor.mean(axis=1)], axis=1)
el_nor.columns = ['month', 'eur_per_MWh']

# Denmark
el_dk = np_df[['DK1', 'DK2']].astype('float')
el_dk = pd.concat([np_df.month, el_dk.mean(axis=1)], axis=1)
el_dk.columns = ['month', 'eur_per_MWh']

# Germany (only 9 months). Only available nine months for Netherlands as well
el_ger = np_df[['DE-LU']].astype('float')
el_ger = pd.concat([np_df.month, el_ger], axis=1)
el_ger.columns = ['month', 'eur_per_MWh']

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
    df = df.astype({'Wtd Avg Price $/MWh':'float', 'Daily Volume MWh':'int'})
    df = df.reset_index(drop=True)
    return df

us_df = import_us()

us_df['Trade Date'] = pd.to_datetime(us_df['Trade Date'])
us_df.sort_values(by='Trade Date')

    # Calculating weighted Average
g = us_df.groupby(by='Trade Date').agg({'Daily Volume MWh':'sum'}).rename(columns={'Daily Volume MWh':'sum_vol'}).reset_index()
merged = pd.merge(us_df, g, on='Trade Date')
weights = merged['Daily Volume MWh'] / merged['sum_vol']
merged['weighted_avg'] = merged['Wtd Avg Price $/MWh'] * weights
el_us = merged.groupby(by='Trade Date').\
    agg({'weighted_avg':'sum'}).reset_index().rename(columns={'Trade Date':'month','weighted_avg':'dollar_per_MWh'})


### Oil and gas
# WTI
wti = pd.read_csv(
    'https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/Cushing_OK_WTI_Spot_Price_FOB.csv',
    skiprows=4,
    header=0,
    names=['month', 'dollars_per_barrel'])

# Brent
brent = pd.read_csv(
    'https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/Europe_Brent_Spot_Price_FOB.csv',
    skiprows=4,
    header=0,
    names=['month', 'dollars_per_barrel'])

# Delete variables in environment not needed in the further analysis
del([merged, pmi_nor_list, pmi_udk, us_filenames, weights, g, np_df, us_df, nordpool_files, pmi_nor_seasadj_list])

# Convert column types
def convert_dtypes(df):
    df = df.astype({'pmi':'float'})
    df['month'] = pd.to_datetime(df['month'])
    return df
pmi_nor = convert_dtypes(pmi_nor)
pmi_nor_seasadj = convert_dtypes(pmi_nor_seasadj)
pmi_dk = convert_dtypes(pmi_dk)
pmi_ger = convert_dtypes(pmi_ger)
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
pmi_dict = {'Norway':pmi_nor,
            'Norway (seas. adj)':pmi_nor_seasadj,
            'Denmark':pmi_dk,
            'Germany':pmi_ger,
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
el_nor