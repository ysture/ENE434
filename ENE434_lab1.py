# Lab 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import geopandas as gpd
import seaborn as sns

oil_fields = pd.read_csv("http://jmaurit.github.io/analytics/labs/data/oil_fields_cross.csv")
oil_fields = oil_fields.dropna()
oil_fields['producing_from']= pd.to_datetime(oil_fields['producing_from'])  # converting dates to datetime
oil_fields.head()

'''
Question 1:
Create a variable invest_per_rec which is investment per million sm3 in recoverable oil (recoverable_oil). 
Plot these variables against producing_from variable and the original recoverable_oil variable. 
How do you interpret the results?
'''

oil_fields.loc[:,'invest_per_rec'] = round(oil_fields['total.invest'] / oil_fields['recoverable_oil'] ,5)
pd.set_option('display.max_rows', 1000)
oil_fields

# plotting 'producing_from' against 'recoverable_oil'
plt.scatter(oil_fields['producing_from'], oil_fields['recoverable_oil'])
plt.ylabel('Investment per recoverable sm3 of oil')
plt.xlabel('Field produced from')
plt.show()

'''
Question 2:
Create a list of the 5 “cheapest” oil fields, that is where the investment is lowest per recoverable oil. 
What do these tend to have in common?
'''
# Consider to use oil_fields[oil_fields['invest_per_rec'] != np.inf] instead of oil_fields
pd.set_option('display.max_columns', 5)
oil_fields.sort_values(by='invest_per_rec').head()

'''
Question 3:
I have a hypothesis that oil fields farther north are more expensive to exploit. 
Explore this hypothesis. Do you think it has merit?
'''
plt.scatter(oil_fields['lat'], oil_fields['invest_per_rec'])
plt.ylabel('Investment per recoverable sm3 of oil')
plt.xlabel('Latitude')
plt.show()

'''
Question 4:
Open-ended question: Accessing and importing data

Actually finding and accessing interesting data you want can be challenging. Importing it into R into the correct format can also be challenging. Here you get a taste of this.

a)
Go to the data portion of the Norwegian Petrioleum Directorate
'''

'''
b)
The tabs at the top indicate the different types of data that is available by level/theme. Try to find some interesting dataset and download it as a .csv file. (Hint, on the left-hand panel, go down to “table view”, then you get a table of data, which you can export by clicking on “Export CSV”).
'''

'''
c)
Once you have downloaded the data, import the data into r using the read_csv() command.
'''
inv = pd.read_csv('https://raw.githubusercontent.com/yggarshan/ENE434/master/field_investment_yearly.csv', error_bad_lines=False)
# remove all observations where investments is zero
inv = inv[inv['prfInvestmentsMillNOK'] != 0]
inv.sort_values(by='prfInvestmentsMillNOK')
'''
d)
If there is a date variable, format that variable as a date (if read_csv() hasn’t automatically done so already)
'''
# inv['prfYear'] = pd.to_datetime(inv['prfYear'], format='%Y')
'''
e)
Plot the data in a meaningful way. Interpret the plot. Is there anything puzling about the data.
'''
df = pd.merge(left=inv, right=oil_fields.loc[:,['name', 'lon','lat']],
               left_on=inv.prfInformationCarrier,
               right_on=oil_fields.name,
               how='left')

df_north = df[df.lat > 63] # north of Bodø
df_mid = df[(df.lat > 60) & (df.lat<=63)] # north of Bergen and south of Bodø
df_south = df[df.lat < 60] # south of Bergen

# Scatterplots of regions
plt.scatter(x=df_north['prfYear'], y=df_north['prfInvestmentsMillNOK'],
            alpha=0.3, c='blue', label='North')
plt.scatter(x=df_mid['prfYear'], y=df_mid['prfInvestmentsMillNOK'],
            alpha=0.3, c='green', label='Mid')
plt.scatter(x=df_south['prfYear'], y=df_south['prfInvestmentsMillNOK'],
            alpha=0.3, c='orange', label='South')
plt.title('Investments at the NCS')
plt.ylabel('Investments')
plt.xlabel('Year')
plt.legend()
plt.show()

# Density plots of regions
sns.distplot(df_north['prfInvestmentsMillNOK'], hist = False, kde = True,
             kde_kws = {'linewidth': 3}, color='blue',
             label = 'North')
sns.distplot(df_mid['prfInvestmentsMillNOK'], hist = False, kde = True,
             kde_kws = {'linewidth': 3}, color='green',
             label = 'Mid')
sns.distplot(df_south['prfInvestmentsMillNOK'], hist = False, kde = True,
             kde_kws = {'linewidth': 3}, color='orange',
             label = 'South')
plt.show()

# Creating map where size of bubbles show size of investments
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
nor = world[world.name == "Norway"]
nor_geom = world.loc[nor_index, 'geometry']

world.plot()
plt.show()
# plot norway
nor.plot()
plt.show()

# plot oil fields
from shapely.geometry import Point, Polygon
import gmaps
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]

gmaps.configure(api_key='AIzaSyBrwck7cRAXXnEPiePL1k5YJnqUstRo0xE')

norway_coord = (67, 15)
fig = gmaps.figure(center=norway_coord, zoom_level=12)
fig.show()

fig = gmaps.figure()
fig

base = nor.plot(color='white', edgecolor='black')
geo_df = gpd.GeoDataFrame(df,
                          crs='crs',
                          geometry=geometry)
geo_df.plot(ax=base)
plt.show()

fig,ax = plt.subplots(figsize = (8,6))
df.plot(ax=base)
plt.show()


'''
LAB2
'''
df = pd.read_csv("https://jmaurit.github.io/anvendt_macro/data/adf.csv")
df.shape
df.head()
df.drop('Unnamed: 0', axis=1, inplace=True)

new_columns= ["working_capital", "working_capital_perc", "fixed_assets", "long_debt",
              "NACE_desc", "NACE_code", "profit", "other_fin_instr", "employees", "depreciation",
              "change_inventories", "operating_income", "operating_costs", "operating_result",
              "equity", "total_assets", "org_type", "principality", "debt", "inv", "cash", "municipality",
              "corp_accounts", "short_debt", "accounts_receivable", "director", "liquidity", "wage_costs",
              "profitability", "current_assets", "pretax_profit", "orgnr", "audit_remarks", "audit_komments",
              "audit_explanation_txt", "audit_explanation", "sales_revenue", "solidity", "status", "founded_date",
              "dividend", "currency_code", "supply_cost", "inventory", "year", "name"]

df.columns = new_columns

# Make sure NACE codes are strings
#df.dtypes
df['NACE_code'] = df['NACE_code'].to_string()
new = list(df.loc[:100,"NACE_code"]).str.split(pat=".", expand=True)
df["NACE_code_1"]= new[0]
df["NACE_code_2"]= new[1]
df