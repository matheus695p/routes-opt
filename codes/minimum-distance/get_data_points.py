import pandas as pd
from src.utils.utils import ls_directory


path_gps = "data/input-data/gps"
files = ls_directory(path=path_gps)

concat = pd.DataFrame()
for file in files:
    print("Leyendo:", file)
    if ".pkl.pkl" in file:
        data = pd.read_pickle(path_gps+"/"+file)
        concat = pd.concat([concat, data], axis=0)

print(concat.shape)
# setiar
concat.reset_index(inplace=True, drop=True)
concat.reset_index(inplace=True)
# hacer el proceso de cleaning con estas larvas
columnas = ['Equipo', 'Date', 'Velocidad', 'Norte', 'Este', 'Cota']
concat = concat[columnas]
concat.rename(columns={"Equipo": "equipo", "Date": "date",
                       "Velocidad": "velocidad", "Este": "este",
                       "Norte": "norte", "Cota": "cota"},
              inplace=True)
# forma mas rápida de filtrar el dataset completo
concat["prefijo"] = concat["equipo"].apply(lambda x: x[0:3])
concat = concat[concat["prefijo"] == "CDH"]
concat.drop(columns=["prefijo"], inplace=True)
concat.reset_index(inplace=True, drop=True)
# guardar el conjunto de puntos
concat.to_pickle(path_gps+"/"+"embeding.pkl")
data = pd.read_pickle(path_gps+"/"+"embeding.pkl")
