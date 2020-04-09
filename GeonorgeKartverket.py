
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
# Funker ikke nå
gyldig = []
import requests
import datetime
url = 'https://ws.geonorge.no/kommunereform/v1/endringer/5033'
r = requests.get(url)
r.status_code
r.reason
j = r.json()
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
    # Add all former municipalitites to list of old municipality numbers if there is an "erstatter" key in the json
    if ('erstattetav' in j['data']):
        new_municnr = j['data']['erstattetav'][0]['id']
        old_municnr = j['data']['kommune']['id']
        try:
            if old_municnr not in gyldigDict[new_municnr]:
                gyldigDict[new_municnr].append(old_municnr)
        except KeyError:
            gyldigDict[new_municnr] = []
    elif ('erstatter' in j['data']):
        new_municnr = j['data']['kommune']['id']
        if len(j['data']['erstatter']) > 1:
            old_municnr = [x['id'] for x in j['data']['erstatter']]
        else:
            old_municnr = j['data']['erstatter'][0]['id']
        try:
            if old_municnr not in gyldigDict[new_municnr]:
                gyldigDict[new_municnr].append(old_municnr)
        except KeyError:
            gyldigDict[new_municnr] = []
    # If there are no registered changes to municipality, then neither 'erstattetav' nor 'erstatter' will be in JSON
    elif (('erstattetav' not in j['data']) and ('erstatter' not in j['data'])):
        current_municnr = j['data']['kommune']['id']
        gyldigDict[current_municnr] = []

end_t = datetime.datetime.now()
print('Time elapsed {}'.format(end_t - start_t))

