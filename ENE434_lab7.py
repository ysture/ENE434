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

# Plotting cost_per_kw
plt.close('all')
fig, ax = plt.subplots()
ax.scatter(pd.to_datetime(pv_df.date), pv_df.cost_per_kw, alpha=0.002, color='black')
plt.ylim(.25e4, 1.24e4)
# Formatting x axis labels
ax.xaxis_date()
plt.show()

# Creating capacity df (showing aggregate trends)
capacity = pv_df.groupby(by='date').agg({'nameplate': 'sum'})
capacity['cumCapacity'] = np.cumsum(capacity)

fig = plt.figure()
plt.plot(capacity.index, capacity.cumCapacity, color='black', linewidth=.75)
plt.show()

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

### Using shape files from kartverket
# files
shape_country = 'norway_shape/NOR_adm_shp/NOR_adm0'
shape_county = 'norway_shape/NOR_adm_shp/NOR_adm1'
shape_princ = 'norway_shape/kartverket/kommuner/kommuner'

shape_country_y = 'C:/Users/Yngve/Google Drive/Skolerelatert/NHH/Master/ENE434/Geodata/Shapefiles/NOR_adm0.shp'
shape_fylke_y = 'C:/Users/Yngve/Google Drive/Skolerelatert/NHH/Master/ENE434/Geodata/Shapefiles/NOR_adm1.shp'
shape_kommune_y = 'C:/Users/Yngve/Google Drive/Skolerelatert/NHH/Master/ENE434/Geodata/Shapefiles/NOR_adm2.shp'


#### kommuner
#Norway boundary
reader_norway = shpreader.Reader(shape_country_y)
norway = reader_norway.geometries()
norway_geom = next(norway)
# Fylker
reader_fylke = shpreader.Reader(shape_fylke_y)
# Kommuner
reader_kommune = shpreader.Reader(shape_kommune_y)
reader_norway.records()

# Create plot
subplot_kw = dict(projection=ccrs.Mercator())

fig, ax = plt.subplots(figsize=(12, 15),
                       subplot_kw=subplot_kw)
ax.set_extent((2, 32.0, 56, 71))
ax.add_geometries(norway_geom, ccrs.PlateCarree(), facecolor="white", edgecolor='black')

'''
# Add fylker
for princ, rec in zip(reader_fylke.geometries(), reader_fylke.records()):
    ax.add_geometries(princ, ccrs.PlateCarree(), facecolor="lightGrey", edgecolor='black', alpha=1)
plt.show()
'''

# Add municipalities (shapefiles, before 2020)
for kommune, rec in zip(reader_kommune.geometries(), reader_kommune.records()):
    if rec.attributes['NAME_2'] in ['Forsand', 'Sandnes']:
        print('Yes')
        color = 'lightgreen'
    else:
        color = 'lightGrey'
    ax.add_geometries([kommune], ccrs.PlateCarree(), facecolor=color, edgecolor='black', alpha=1)
fig.tight_layout()
plt.show()

# Add municipalities (POSTgis, current from 2020)
import fiona
import geopandas as gpd
from osgeo import ogr

gdb_file = 'C:/Users/Yngve/Google Drive/Skolerelatert/NHH/Master/ENE434/Geodata/Basisdata_0000_Norge_3035_Kommuner_FGDB'
# Get all the layers from the .gdb file
driver = ogr.GetDriverByName(gdb_file)
layers = fiona.listlayers(gdb_file)

for layer in layers:
    gdf = gpd.read_file(gdb_file,layer=layer)
    # Do stuff with the gdf

# GeoNorge API for kommunereform
# Create df of all municipality changes
import requests
import datetime
url = 'https://ws.geonorge.no/kommunereform/v1/endringer/0301'
r = requests.get(url)
r.status_code
r.reason
j = r.json()
gyldig = []
gyldigDict = {}
start_t = datetime.datetime.now()
for i in range(0,10000):
    i = '{:0>4}'.format(i) # Add a leading zero if number has three or less characters
    try:
        r = requests.get(url + i)
        j = r.json()
    except Exception as e:
        print(i, e)
    try:
        current_municnr = j['data']['kommune']['id']
        current_municname = j['data']['kommune']['navn']
    except TypeError:
        continue
    # Add municipality number to gyldig-list if there are no "erstattetav" keys in the json
    if 'erstattetav' not in j['data']:
        obs = (current_municname, current_municnr)
        if obs not in gyldig:
            gyldig.append(obs)
    # Add all former municipalitites to list of old municipality numbers if there is an "erstatter" key in the json
    elif ('erstattetav' in j['data']):
        new_municnr = j['data']['erstattetav'][0]['id']
        old_municnr = j['data']['kommune']['id']
        try:
            oldMunicList = gyldigDict[new_municnr]
        except KeyError:
            gyldigDict[new_municnr] = []
        if old_municnr not in gyldigDict[new_municnr]:
            gyldigDict[new_municnr].append(old_municnr)
    # If there are no registered changes to municipality, then neither 'erstattetav' nor 'erstatter' will be in JSON
    elif (('erstattetav' not in j['data']) and ('erstatter'  not in j['data'])):
        current_municnr = j['data']['kommune']['id']
        gyldigDict[current_municnr] = []
end_t = datetime.datetime.now()
print('Time elapsed {}'.format(end_t - start_t))

