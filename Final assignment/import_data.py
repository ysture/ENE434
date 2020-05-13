'''File to import data used in final ENE434 Assignment
Mangler: Alle PMI-tall samt gasspriser.
'''
import pandas as pd
import numpy as np
from tabula import read_pdf
import os
import re

#TODO importer europeiske el-priser
#TODO importer amerikanske el-priser og ta gjennomsnitt av disse

### PMI
# Germany
pmi_ger = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/germany.markit-manufacturing-pmi.csv',
                      sep='\t',
                      usecols=[0,1],
                      header=0,
                      names=['month', 'index_value'])

## Norway
# Ordinary
pmi_nor_list = read_pdf(
    'C:\\Users\\Yngve\\Google Drive\\Skolerelatert\\NHH\\Master\\ENE434\\Final assignment\\input\\PMI_norway_pdf.pdf',
    pages=[1, 2, 3, 4, 5], multiple_tables=False,
    pandas_options={'header': None,
                    'names': ['PMI', 'Production', 'New orders', 'Employment',
                              'Suppliers Delivery Time', 'Inventory of Purchased Goods']})

# Seasonally adjusted
pmi_nor_seasadj_list = read_pdf(
    'C:\\Users\\Yngve\\Google Drive\\Skolerelatert\\NHH\\Master\\ENE434\\Final assignment\\input\\PMI_norway_pdf.pdf',
    pages=[6, 7, 8, 9, 10], multiple_tables=False,
    pandas_options={'header': None,
                    'names': ['PMI', 'Production', 'New orders', 'Employment',
                              'Suppliers Delivery Time', 'Inventory of Purchased Goods']})


pmi_nor = pmi_nor_list[0].reset_index().rename(columns={'index':'month'}).dropna()
pmi_nor_seasadj = pmi_nor_seasadj_list[0].reset_index().rename(columns={'index':'month'}).dropna()

# UK
pmi_uk = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/united-kingdom.markit-manufacturing-pmi.csv',
                     sep='\t',
                     usecols=[0,1],
                     header=0,
                     names=['month', 'index_value'])

## Denmark and US
pmi_udk = pd.read_csv('https://raw.githubusercontent.com/ysture/ENE434/master/Final%20assignment/input/us_dk_pmi.csv',
                      sep=';',
                      header=0,
                      names=['month', 'us_index', 'dk_index'])

# US
pmi_us = pmi_udk[['month', 'us_index']]

# Denmark
pmi_dk = pmi_udk[['month', 'dk_index']].dropna().reset_index()

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

el_nor = np_df[['Oslo', 'Kr.sand', 'Bergen', 'Molde', 'Tr.heim', 'Tromsø']]
# Average
el_nor.agg('avg')
el_nor.mean(axis=1)


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
