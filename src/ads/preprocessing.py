import numpy as np


def gps_preprocessing(df):
    """
    Hacer preprocesamiento con los datos de gps, encontrando las estimaciones
    de velocidad, angulo de inclinación con estado de inclinación
    Parameters
    ----------
    df : dataframe
        gps data
    Returns
    -------
        gps-preprocessed.
        unidades de gps
        aceleración: [m/s2]
        angulo: [grados]
        velocidad: [km/hr]
    """
    velocidad_maxima = 60

    # sacar últimos estados
    df = axis_time_movement(df)
    df = last_state_xyz(df)
    # calcular movimiento en metros
    df["movimiento"] = df.apply(
        lambda x: euclidian_distance(
            x["diff_este"], x["diff_norte"], x["diff_cota"]), axis=1)
    # calcular velocidad estimada con gps
    df["velocidad_estimada"] = df["movimiento"] / df["diff_date"] * 3.6
    # filtrar aquellos valores de velocidad por sobre vel_max
    df["velocidad_estimada"] = df["velocidad_estimada"].apply(
        lambda x: velocidad_maxima if x > velocidad_maxima else x)
    # cálculo de la aceleración
    df["diff_vel_estimada"] = df.sort_values(['equipo', 'date']).\
        groupby('equipo')['velocidad_estimada'].diff()
    # aceleración en metros por segundo
    df["aceleracion"] = (df["diff_vel_estimada"] / 3.6) / df["diff_date"]
    # aceleración corregida a solo un valor positivo
    df["aceleracion_positiva"] = df["aceleracion"].apply(
        lambda x: 0 if x < 0 else x)
    df["aceleracion_negativa"] = df["aceleracion"].apply(
        lambda x: 0 if x > 0 else x)
    # calcular angulo de inclinación
    df["angulo"] = np.arcsin(
        df["diff_cota"] / df["movimiento"]) * 180 / np.pi
    # angulo en radianes
    df["angulo_rad"] = np.arcsin(
        df["diff_cota"] / df["movimiento"])
    # rellenar valores con cero, aquellos donde el movimiento fue cero
    df["angulo"].fillna(value=0, inplace=True)
    df["angulo_rad"].fillna(value=0, inplace=True)
    # angulo positivo y negativo
    df["angulo_positivo"] = df["angulo"].apply(lambda x: 0 if x < 0 else x)
    df["angulo_negativo"] = df["angulo"].apply(lambda x: 0 if x > 0 else x)
    # etiquetar el estado de la pendiente
    df["estado_pendiente"] = df["angulo"].apply(lambda x: slope_state(x))
    # sacar dataframe de salida
    df.sort_values(by=["equipo", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def last_state_xyz(df):
    """
    Obtiene el último estado (x, y, z and speed) de cada evento de gps
    Parameters
    ----------
    df : dataframe
        gps data
    Returns
    -------
        4 columnas adicionales al estado
    """
    df['prev_este'] = df.sort_values(by=["equipo", "date"]).\
        groupby("equipo").shift(1)["este"]
    df['prev_norte'] = df.sort_values(by=["equipo", "date"]).\
        groupby("equipo").shift(1)["norte"]
    df['prev_cota'] = df.sort_values(by=["equipo", "date"]).\
        groupby("equipo").shift(1)["cota"]
    df['prev_vel'] = df.sort_values(by=["equipo", "date"]).\
        groupby("equipo").shift(1)["velocidad"]
    return df


def euclidian_distance(diff_x, diff_y, diff_z):
    """
    Determinar la distancia euclidiana entre el punto anterior y el actual
    de forma determinar la distancia en metros.
    Parameters
    ----------
    diff_x : float64
        diferencia en la coordenada x.
    diff_y : float64
        diferencia en la coordenada y.
    diff_z : float64
        diferencia en la coordenada z.
    Returns
    -------
    distance : float64
        movimiento discreto entre un punto x e x+1, donde x+1 es el registro
        siguiente.
    """
    distance = np.abs(
        diff_x*diff_x) + np.abs(diff_y*diff_y) + np.abs(diff_z*diff_z)
    distance = np.sqrt(distance)
    return distance


def axis_time_movement(df):
    """
    Calcula el movimiento x,y e z en un delta T de movimiento
    Parameters
    ----------
    df : Dataframe
        gps data
    Returns
    -------
        gps data with movement on axis and time
    """
    df["diff_date"] = df.sort_values(['equipo', 'date']).\
        groupby('equipo')['date'].diff().dt.total_seconds()
    df["diff_norte"] = df.sort_values(['equipo', 'date']).\
        groupby('equipo')['norte'].diff()
    df["diff_este"] = df.sort_values(['equipo', 'date']).\
        groupby('equipo')['este'].diff()
    df["diff_cota"] = df.sort_values(['equipo', 'date']).\
        groupby('equipo')['cota'].diff()
    return df


def slope_state(angulo, lim=4):
    """
    En función de la pendiente que va avanzando el camión, se define un estado
    de inclinación a definir en función de la distribución de cada mina
    Parameters
    ----------
    angulo : int
        angulo en grados.
    lim : int, optional
        limite de angulo para ser considerado en subida. The default is 4.
    Returns
    -------
    estado : TYPE
        retorna el estado en función del grado de inclinación, plano, subiendo
        o bajando.
    """
    if angulo > lim:
        estado = "subiendo"
    elif angulo < -lim:
        estado = "bajando"
    else:
        estado = "plano"
    return estado


def interquartile_range(df, column="angulo"):
    """
    Filtrar el dataframe para hacer detección de anomalias en a través
    del método IQR, y filtrar los ángulos que fueron anomalos

    Parameters
    ----------
    df : dataframe
        dataframe de pendientes entre trayectos.
    column : string, optional
        nombre de la columna a filtrar. The default is "angulo".

    Returns
    -------
    df : dataframe
        dataframe de pendientes entre trayectos, con los angulos filtrados.

    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - iqr * 1.5
    upper_limit = q1 + iqr * 1.5
    df = df[(df[column] > lower_limit) & (df[column] > upper_limit)]
    df.reset_index(drop=True, inplace=True)
    return df
