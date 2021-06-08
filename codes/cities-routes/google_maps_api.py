import json
import googlemaps
import pandas as pd


# cargar la data de ciudadeds chilenas
data = pd.read_csv("data/cities-of-chile/ciudades_de_chile.csv")

# api key y cliente de google maps
# lectura del api key
with open('keys.json') as json_file:
    dict_ = json.load(json_file)
    API_KEY = dict_["google_maps_api"]
# cliente de google maps
gmaps = googlemaps.Client(key=API_KEY)

# filtrar las columnas
data = data[['city', 'admin_name', 'lat', 'lng']]
data["city"] = data["city"] + " / " + data["admin_name"]
data.drop(columns=["admin_name"], inplace=True)

# sacar la información para todos
for ii in range(len(data)):
    city1 = data["city"].iloc[ii]
    lat1 = data["lat"].iloc[ii]
    long1 = data["lng"].iloc[ii]
    for jj in range(len(data)):
        city2 = data["city"].iloc[jj]
        lat2 = data["lat"].iloc[jj]
        long2 = data["lng"].iloc[jj]
        # array de latitudes y longitures
        origins = (lat1, long1)
        destination = (lat2, long2)
        if city1 == city2:
            pass
        else:
            result = gmaps.distance_matrix(origins, destination,
                                           mode='driving')
