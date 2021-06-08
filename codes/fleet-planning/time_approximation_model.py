import warnings
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
# from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from src.fleet.evaluation_metrics import rmsle
from src.fleet.preparation import nn_preparation, inter_quantile_range
from src.fleet.config_nn import arguments_parser
from src.fleet.models import create_nn, create_cnn1d
from src.fleet.visualizations import (training_history,
                                      plot_multiple_xy_results,
                                      kernel_density_estimation)
# from src.fleet.loss_functions import handler_routes
warnings.filterwarnings("ignore")

# argumentos
args = arguments_parser()
filename = "cnn_nn"

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
x_nn_train, x_nn_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=args.seed)

# convertir a logaritmo el y target
# y_train = np.log(y_train+1)

# reshapear para las capas convolucionales
x_cnn1d_train = x_nn_train.reshape(
    x_nn_train.shape[0], x_nn_train.shape[1], 1)
x_cnn1d_test = x_nn_test.reshape(
    x_nn_test.shape[0], x_nn_test.shape[1], 1)

# crear las redes neuronales
nn = create_nn(x_nn_train.shape[1])
cnn1d = create_cnn1d((x_cnn1d_train.shape[1], 1))

# combinar las entradas
combined_input = tf.keras.layers.concatenate([nn.output, cnn1d.output])

# continuar la concatenación de la red
x = tf.keras.layers.Dense(256)(combined_input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
# x = tf.keras.layers.Dropout(0.2)(x)

# x = tf.keras.layers.Dense(64)(combined_input)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)
# x = tf.keras.layers.Dropout(0.2)(x)

# x = tf.keras.layers.Dense(32, activation="relu")(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dense(y_train.shape[1], activation="linear")(x)

# generación del modelo siames
model = tf.keras.models.Model(inputs=[nn.input, cnn1d.input], outputs=x)
print(model.summary())

# tf.keras.losses.handler_loss_function = handler_routes

# compilar el modelo
# model.compile(loss=handler_routes(factor=1),
#               optimizer=args.optimizer)
model.compile(loss=args.loss,
              optimizer=args.optimizer)


# llamar callbacks de early stopping
tf.keras.callbacks.Callback()
stop_condition = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=args.patience,
    verbose=1,
    min_delta=args.min_delta,
    restore_best_weights=True)

# bajar el learning_rate durante la optimización
learning_rate_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=args.lr_factor,
    patience=args.lr_patience,
    verbose=1,
    mode="auto",
    cooldown=0,
    min_lr=args.lr_min)

# cuales son los callbacks que se usaran
callbacks_ = [stop_condition, learning_rate_schedule]

# entrenar
history = model.fit(x=[x_nn_train, x_cnn1d_train], y=y_train,
                    validation_split=args.validation_size,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    shuffle=True,
                    verbose=1,
                    callbacks=callbacks_)

# plot de historial de entrenamiento
training_history(history, model_name="all", filename="Model")

# resultados del modelo np.log(y_train+1)
results = model.evaluate([x_nn_test, x_cnn1d_test],  y_test,
                         batch_size=args.batch_size)

# predictions
y_pred = model.predict(x=[x_nn_test, x_cnn1d_test])
# desconvertir
# y_pred = np.exp(y_pred) - 1
print(y_pred.shape, y_test.shape)
ind = 0
# plotear multiples resultados
plot_multiple_xy_results(y_pred, y_test, target, ind,
                         folder_name=f"{filename}")
# guardar el modelo
model.save(f"models/{filename}.h5")
