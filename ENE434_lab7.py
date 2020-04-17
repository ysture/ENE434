import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

pv_df = pd.read_csv("http://jmaurit.github.io/analytics/labs/data/pv_df.csv", low_memory=False)

# We are only going to investigate production between 2006 and 2015
pv_df = pv_df[(pv_df.year < 2015) & (pv_df.year > 2006)]
pv_df['monthdate'] = pd.to_datetime((pv_df['month'].astype(str) + '-' + pv_df['year'].astype(str)), format='%m-%Y')
pv_df['date'] = pd.to_datetime(pv_df.date)
pv_df.sort_values(by='date', inplace=True)
print('Zero print')
# Plotting cost_per_kw
plt.close('all')
fig, ax = plt.subplots()
ax.scatter(pd.to_datetime(pv_df.date), pv_df.cost_per_kw, alpha=0.002, color='black')
plt.ylim(.25e4, 1.24e4)
# Formatting x axis labels
ax.xaxis_date()
plt.show()
plt.close()

print('First print')
# Creating capacity df (showing aggregate trends)
capacity = pv_df.groupby(by='date').agg({'nameplate': 'sum'})
capacity['cumCapacity'] = np.cumsum(capacity)

fig = plt.figure()
plt.plot(capacity.index, capacity.cumCapacity, color='black', linewidth=.75)
plt.show()
print('Second print')
# We’ll aggregate both cost and cost less subsidy to the month-year average.
cost = pv_df.groupby(by='monthdate').agg({'cost_per_kw': 'mean',
                                          'cost_ex_subsid_per_kw': 'mean'})
# Rename columns
cost.rename(columns={'cost_per_kw': 'avgCost', 'cost_ex_subsid_per_kw': 'avgCost_less_sub'}, inplace=True)

# Plot avg cost and avgCost_less_sub
fig = plt.figure()
plt.plot(cost.avgCost)
plt.plot(cost.avgCost_less_sub)
plt.title('Average cost in $ per KWh in California')
plt.ylabel('$ / KWh')
plt.legend()
plt.show()
print('Third print')
'''
# Mapping the data
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import geopandas as gpd

proj = ccrs.Mercator()
coord_1 = (58.745, 5.804)
coord_2 = (70.9598, 25.380)

# Create simple Norway map
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent((-15, 35, 55, 70))
# ax.set_global()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.plot([coord_1[1], coord_2[1]], [coord_1[0], coord_2[0]], transform=ccrs.Geodetic())
fig.tight_layout()
plt.show()

# Add fylker
kw = dict(resolution='10m', category='cultural',
          name='admin_1_states_provinces')
states_shp = shpreader.natural_earth(**kw)
shp = shpreader.Reader(states_shp)

# Major Norwegian cities
cities = ["Bergen", "Oslo", "Trondheim", "Stavanger", "Kristiansand", "Tromsø"]
# (long, lat)
coords = [[5.3221, 60.3913], [10.7522, 59.9139], [10.3951, 63.4305], [5.7331, 58.9700],
          [8.0182, 58.1599], [18.9553, 69.6492]]
city_coords = zip(cities, coords)

# Add visualizations (Major Norwegian Cities)
subplot_kw = dict(projection=ccrs.Mercator())

fig, ax = plt.subplots(figsize=(12, 15),
                       subplot_kw=subplot_kw)
ax.set_extent((2, 32.0, 56, 71))
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)
for record, state in zip(shp.records(), shp.geometries()):
    if record.attributes['admin'] == 'Norway':
        ax.add_geometries([state], ccrs.PlateCarree(), facecolor="lightGray", edgecolor='black')
for city, coords in city_coords:
    plt.plot(coords[0], coords[1], color='red', transform=ccrs.Geodetic(), marker='o', markersize=10)
    plt.text(coords[0] - 0.5, coords[1], city, horizontalalignment='right', transform=ccrs.Geodetic(), color='red')
fig.tight_layout()
plt.show()

# Plot wind power installations in California
# plt.rcParams['agg.path.chunksize'] = 5000000000000000000000000000000000000000
pv_df_coords = pv_df[['longitude', 'latitude']].dropna()
subplot_kw = dict(projection=ccrs.Mercator())
fig, ax = plt.subplots(figsize=(12, 15),
                       subplot_kw=subplot_kw)
for record, state in zip(shp.records(), shp.geometries()):
    if record.attributes['gn_a1_code'] == 'US.CA':
        add_to_border = 0.5
        coords_map = state.bounds
        ax.set_extent((coords_map[0] - add_to_border, coords_map[2] + add_to_border, coords_map[1] - add_to_border,
                       coords_map[3] + add_to_border))
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND)
        ax.add_geometries([state], ccrs.PlateCarree(), facecolor="lightGray", edgecolor='black')
# Plot installations
plt.plot(pv_df_coords.longitude, pv_df_coords.latitude, 'ro', alpha=0.01, transform=ccrs.Geodetic(),
         markersize=12)
fig.tight_layout()
plt.show()


# Investigation of Natural Earth (https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/) data
# Remember to place [] around geometry object when plotting maps using shapesfiles and cartopy
def print_regions_and_provinces(countries):
    for record in shp.records():
        country = record.attributes['admin']
        region_sub = record.attributes['region_sub']
        province_name = record.attributes['name']
        if country in countries:
            print(country, region_sub, province_name)


print_regions_and_provinces('Norway')
print_regions_and_provinces('Denmark')
print_regions_and_provinces('Sweden')
print_regions_and_provinces('United States of America')
'''
# Estimating a learning curve
# First we need to create a new variable representing the cumulative capacity (since 2006).
cumsum_cap = pv_df.sort_values(by='date').agg({'nameplate': 'cumsum'})
pv_df['cum_cap'] = cumsum_cap

# Formally estimate linear model between the two
# Remove zero-values of cost_per_kw as log of these are infeasible
pv_df = pv_df[pv_df.cost_per_kw != 0].copy()
pv_df['log2_cum_cap'] = np.log2(pv_df.cum_cap)
pv_df['log2_cost_per_kw'] = np.log2(pv_df.cost_per_kw)

# Create linear model
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

Y = pv_df.loc[:, 'log2_cost_per_kw']
X = pv_df.loc[:, 'log2_cum_cap']
X_ = sm.add_constant(X)
model = sm.OLS(Y, X_)
results = model.fit()
ypred = results.predict()
results.summary()

print('Fourth print')


# Plot polynomial fit
polynomial_features = PolynomialFeatures(degree=4)
xp = polynomial_features.fit_transform(X_)
model = sm.OLS(Y, xp).fit()
ypred_poly = model.predict(xp)

# Plot lm fit and x4 polynomial fit
fig = plt.figure()
plt.scatter(np.log2(pv_df.cum_cap), np.log2(pv_df.cost_per_kw), c='black', alpha=0.1)
plt.plot(X, ypred, label='LM fit')
plt.plot(X, ypred_poly, label='Polynomial fit')
plt.legend()
plt.show()
print('Fifth print')

# Exercise 1.)
    # Estimate separate learning curve pre-2012 and post-2012. Can you do this with a single regression?
    # What would be the advantages and disadvantages of doing so?
# Creating column denoting if an observation is pre or post 2012
pv_df['Period'] = [0 if x.year < 2012 else 1 for x in pv_df.monthdate]
# Creating dummy variables
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
ohe = OneHotEncoder()
period_encoded = ohe.fit_transform(pv_df.Period.values.reshape(-1,1)).toarray()
ohe.categories_ # Displays all categories in the categorical columns that are now encoded
# Add dummy columns back to original data frame
df_period_encoded = pd.DataFrame(period_encoded, columns = ["Period_"+str(int(i)) for i in range(period_encoded.shape[1])])
pv_df_encoded = pd.concat([pv_df, df_period_encoded], axis=1)
pv_df_encoded = pv_df_encoded[~pv_df_encoded.Period_0.isna()]
# Create simple linear regression model of the data (without test and train data)
y = pv_df_encoded['log2_cost_per_kw'].dropna()
x = pv_df_encoded[['log2_cum_cap', 'Period_0', 'Period_1']].dropna()
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
ypred = results.predict()
results.summary()
print(results.summary())
print('Sixth print')


# Exercise 2.)
    # Estimate the relationship between cumulative capacity and solar power costs with a local linear regression, or LOESS.
    # (See section 7.6 and 7.82 in ISL). How is local linear regression similar to and different from splines.
    #  Use local linear regression create a point forecast of costs from 2015 to 2020. Does this suffer from
    #  the same problems as the Spline?
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from skmisc.loess import loess
import pylab as pylab

# Inspired by stackoverlow
l = loess(X,Y)
l.fit()
a = 100000
X_loess = X[:a]
Y_loess = Y[:a]
pred_loess = l.predict(X_loess)
lowess = pred_loess.values
pylab.plot(X_loess, Y_loess, '+', color='blue')
pylab.plot(X_loess, lowess, color='orange')

# Adding confidence intervals crashes the process for some reason. Can bootstrapping be used to find st.errors?
#pred_loess = l.predict(X[:a], stderror=True)
#conf_loess = pred_loess.confidence()
#ll = conf_loess.lower
#ul = conf_loess.upper
#pylab.fill_between(X[:a],ll,ul,alpha=.33)
pylab.show()


# Plot lm fit, x4 polynomial fit and LOESS
fig = plt.figure()
plt.scatter(np.log2(pv_df.cum_cap), np.log2(pv_df.cost_per_kw), c='black', alpha=0.1)
plt.plot(X, ypred, label='LM fit')
plt.plot(X, ypred_poly, label='Polynomial fit')
plt.plot(X_loess, lowess, label='LOESS')
plt.legend()
plt.show()