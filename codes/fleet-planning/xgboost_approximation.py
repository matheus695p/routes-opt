import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
# from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.fleet.evaluation_metrics import rmsle
from src.fleet.preparation import nn_preparation, inter_quantile_range
from src.fleet.config_nn import arguments_parser
from src.fleet.visualizations import (plot_multiple_xy_results,
                                      kernel_density_estimation)
warnings.filterwarnings("ignore")

# argumentos
args = arguments_parser()
filename = "xgboost"

# data de entrenamiento
sample_df = pd.read_csv("data/fleet-data/train.csv")
print("dimensiones de la data:", sample_df.shape)
print("Columnas:", list(sample_df.columns))
print(sample_df["store_and_fwd_flag"].value_counts())
head = sample_df.head(100)
print(head)

# convertir caracteristicas categoricas
sample_df["store_and_fwd_flag"] = sample_df["store_and_fwd_flag"].apply(
    lambda x: 0 if x == "N" else 1)

# convertir a datetime
sample_df["dropoff_datetime"] = pd.to_datetime(
    sample_df["dropoff_datetime"], format='%Y-%m-%d %H:%M:%S')
sample_df["pickup_datetime"] = pd.to_datetime(
    sample_df["pickup_datetime"], format='%Y-%m-%d %H:%M:%S')

# crear más caracteriticas al modelo
sample_df["pickup_month"] = sample_df["pickup_datetime"].dt.month
sample_df["pickup_day"] = sample_df["pickup_datetime"].dt.day
sample_df["pickup_weekday"] = sample_df["pickup_datetime"].dt.weekday
sample_df["pickup_hour"] = sample_df["pickup_datetime"].dt.hour
sample_df["pickup_minute"] = sample_df["pickup_datetime"].dt.minute

# Diferencias entre latitude y longitude
sample_df["latitude_difference"] = sample_df["dropoff_latitude"] - \
    sample_df["pickup_latitude"]
sample_df["longitude_difference"] = sample_df["dropoff_longitude"] - \
    sample_df["pickup_longitude"]

# duración del viaje en minutos
sample_df["trip_duration"] = sample_df["trip_duration"] / 60

# calcular la distancia
sample_df["trip_distance"] = 0.621371 * 6371 *\
    (abs(2 * np.arctan2(
        np.sqrt(np.square(
            np.sin((
                abs(sample_df["latitude_difference"]) * np.pi / 180) / 2))),
        np.sqrt(1-(np.square(
            np.sin((abs(
                sample_df["latitude_difference"]) * np.pi / 180) / 2)))))) +
     abs(2 * np.arctan2(
         np.sqrt(np.square(
             np.sin((abs(
                 sample_df["longitude_difference"]) * np.pi / 180) / 2))),
         np.sqrt(1-(
             np.square(
                 np.sin(
                     (abs(sample_df[
                         "longitude_difference"]) * np.pi / 180) / 2)))))))

# modelar
sample_df.drop(["id", "vendor_id", "pickup_datetime",
                "dropoff_datetime"], inplace=True, axis=1)
kernel_density_estimation(sample_df, "trip_duration", name="all")
sample_df = inter_quantile_range(sample_df, "trip_duration")
kernel_density_estimation(sample_df, "trip_duration", name="all")

# columnas target y columnas objectivo
target = ["trip_duration"]
columns = list(sample_df.columns)
# target vs features
x, y = nn_preparation(sample_df, target)

# normalizar los dataos entre 0,1
sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)
joblib.dump(sc, f"models/{filename}.save")

# división del conjunto de datos
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=args.seed)

# división para obtener los de validación
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.25, random_state=args.seed)

# hiperparametros
nrounds = 2000
params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'learning_rate': 0.05,
          'max_depth': 14,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'silent': 1,
          'feval': 'rmsle'}

# test de train y validación
dtrain = xgb.DMatrix(x_train, np.log(y_train+1))
dval = xgb.DMatrix(x_val, np.log(y_val+1))

# para mirar los errores de val y train
watchlist = [(dval, 'eval'), (dtrain, 'train')]

# modelo
gbm = xgb.train(params,
                dtrain,
                num_boost_round=nrounds,
                evals=watchlist,
                verbose_eval=True)

pred = np.exp(gbm.predict(xgb.DMatrix(x_test))) - 1
y_pred = pred.reshape(pred.shape[0], -1)


abs_error = pd.DataFrame(abs(y_pred - y_test), columns=["count"])[
    "count"].value_counts(bins=30)
abs_error = pd.DataFrame(abs_error)
abs_error["porcentaje"] = abs_error["count"] / abs_error["count"].sum() * 100
abs_error.reset_index(drop=False, inplace=True)
abs_error.rename(columns={"index": "rango error"}, inplace=True)
abs_error["acumulativo"] = abs_error["porcentaje"].cumsum()

print("Error absoluto medio: ", (abs(y_pred - y_test)).mean())

# indice
ind = 0
# plotear multiples resultados
plot_multiple_xy_results(y_pred, y_test, target, ind,
                         folder_name=f"{filename}")

feature_scores = gbm.get_fscore()
summ = 0
for key in feature_scores:
    summ = summ + feature_scores[key]

for key in feature_scores:
    feature_scores[key] = feature_scores[key] / summ * 100
print(feature_scores)


filename = f"models/{filename}.sav"
pickle.dump(gbm, open(filename, 'wb'))
