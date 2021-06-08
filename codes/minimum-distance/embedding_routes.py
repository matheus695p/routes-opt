import numpy as np
import pandas as pd

path_gps = "data/input-data/gps"
data = pd.read_pickle(path_gps+"/"+"embeding.pkl")
head = data.head(1000)
# pasar a enteros
data["norte"] = data["norte"].astype(int)
data["este"] = data["este"].astype(int)
data["cota"] = data["cota"].astype(int)

# Calculamos la diferencia de tiempo
data["diff"] = data.sort_values(by=["equipo", "date"]).\
    groupby("equipo")["date"].diff()
data["diff"] = data["diff"].dt.total_seconds()
# filtrar por todos aquellos que tengan interconectivdad
data = data[data["diff"] < 60*4]

# Se revisa el punto anterior
data["norte_n"] = data.sort_values(by=["equipo", "date"]).\
    groupby("equipo")["norte"].shift()
data["este_n"] = data.sort_values(by=["equipo", "date"]).\
    groupby("equipo")["este"].shift()
data["cota_n"] = data.sort_values(by=["equipo", "date"]).\
    groupby("equipo")["cota"].shift()

# Creamos un dataframe de points conectados
points = data.groupby(by=["norte", "este", "cota", "norte_n",
                          "este_n", "cota_n"]).count()["equipo"]
points = pd.DataFrame(points).reset_index()
points = points.round(-1)
points["distancia"] = np.sqrt((points["norte_n"] - points["norte"])**2 +
                              (points["este_n"] - points["este"])**2 +
                              (points["cota_n"] - points["cota"])**2)

# Eliminamos la conexion con si mismo
points = points[points["distancia"] != 0]

# Eliminamos la cota 0
points = points[points["cota"] != 0]
points = points[points["cota_n"] != 0]
points = pd.DataFrame(points).reset_index()

# Formateamos el dataframe
points.drop_duplicates(inplace=True)
points.dropna(inplace=True)
points.reset_index(drop=True, inplace=True)
points = points.astype(int)

# Filtramos las distancias mayores a 55 metros
points = points[points["distancia"] < 55]
points.reset_index(drop=True, inplace=True)

# Damos nombre a los points
points["origen"] = (points["norte"].astype(str) + "-" +
                    points["este"].astype(str) + "-" +
                    points["cota"].astype(str))
points["destino"] = (points["norte_n"].astype(str) + "-" +
                     points["este_n"].astype(str) + "-" +
                     points["cota_n"].astype(str))

# solo guardar los puntos
points = points[['distancia', 'origen', 'destino']]
points.drop_duplicates(inplace=True)
points.reset_index(drop=True, inplace=True)

# guardar el embedding de puntos
points.to_pickle(path_gps+"/"+"gps_routes_embeding.pkl")
