import warnings
import pandas as pd
from sklearn.utils import shuffle
from src.optimization.shortest import create_list_points, shortest_path
from src.ads.cleaning import convert_df_float
from src.utils.visualizations import vista_eje_z
warnings.filterwarnings("ignore")

# path de gps
path_gps = "data/input-data/gps"
# número de puntos a evaluar
number_points = 100

# traer puntos originales
original_points = pd.read_pickle(path_gps+"/"+"embeding.pkl")
original_points = original_points[['norte', 'este', 'cota']]
original_points = shuffle(original_points, random_state=21)
original_points.reset_index(drop=True, inplace=True)
original_points = original_points.iloc[0:100000, :]

# traer la data del embedding
points = pd.read_pickle(path_gps+"/"+"gps_routes_embeding.pkl")
# puntos en una lista
points_set = []
points.reset_index(inplace=True, drop=True)
points.apply(create_list_points, axis=1, args=(points_set,))

# puntos para procesar
points_to_process = shuffle(points, random_state=21)[0:number_points]
points_to_process = points_to_process[["origen"]]
points_to_process.reset_index(inplace=True, drop=True)
points_to_process.rename(columns={"origen": "puntos"}, inplace=True)

# puntos actuales
teta = int(number_points / 2)
actual_points = points_to_process.iloc[0:teta, :]
actual_points.reset_index(inplace=True, drop=True)

# puntos target en el camino
target_points = points_to_process.iloc[teta:number_points, :]
target_points.reset_index(inplace=True, drop=True)

# Encontrar caminos más cortos
for ind in range(len(actual_points)):
    print("Trayecto: ", ind)
    pt1 = actual_points["puntos"].iloc[ind]
    pt2 = target_points["puntos"].iloc[ind]
    # caminos
    track = shortest_path(points_set, pt1, pt2)
    track = track[1]
    track = pd.DataFrame(track, columns=["puntos_camino"])
    track[["norte", "este", "cota"]] = track["puntos_camino"].apply(
        lambda x: pd.Series(str(x).split("-")))
    track = convert_df_float(track)
    # mostrar el camino trazado
    vista_eje_z(original_points, track,
                filename=f"images/paths/path_z_{str(ind)}")
