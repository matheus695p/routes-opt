import warnings
import pandas as pd
from src.ads.cleaning import (clean_gps, clean_loads)
from src.ads.preprocessing import gps_preprocessing
warnings.filterwarnings("ignore")

# traerme la data de loads
loads = pd.read_pickle("data/raw-data/loads/loads-2020-07-15.pkl")
loads = clean_loads(loads)

# traerme la data de gps
gps = pd.read_pickle("data/raw-data/gps/gps-2020-07-15.pkl")
gps = clean_gps(gps)
gps = gps_preprocessing(gps)

# dataframe para trabajar con gps
gps_clean = gps[['equipo', 'date', 'velocidad', 'velocidad_estimada',
                 'movimiento', 'aceleracion', 'aceleracion_positiva',
                 'aceleracion_negativa', 'angulo', 'angulo_rad',
                 'angulo_positivo', 'angulo_negativo', 'estado_pendiente']]
gps_clean.dropna(inplace=True)
gps_clean.reset_index(drop=True, inplace=True)

# puntos de origen y destino únicos
loads["trayecto"] = loads["origen"] + " / " + loads["destino"]

# hacer el dataframe de la distribución de caminos
path_distribution = pd.DataFrame()
# iterar para cada indice
for ind in range(len(loads)):
    equipo = loads["equipo"].iloc[ind]
    trayecto = loads["trayecto"].iloc[ind]
    fecha_i = loads["date"].iloc[ind]
    fecha_f = loads["fecha_fin_descarga"].iloc[ind]
    # filtro de gps
    gps_ii = gps_clean[(gps_clean["date"] >= fecha_i) &
                       (gps_clean["date"] <= fecha_f) &
                       (gps_clean["equipo"] == equipo)]
    gps_ii.reset_index(drop=True, inplace=True)
    # gps
    gps_ii = gps_ii[['equipo', 'velocidad', 'movimiento', 'aceleracion',
                     'angulo', 'estado_pendiente']]
    gps_ii["trayecto"] = trayecto
    path_distribution = pd.concat([path_distribution, gps_ii], axis=0)

path_distribution.reset_index(drop=True, inplace=True)
path_distribution.to_pickle("data/raw-data/gps/angle_distribution.pkl")
