import pytz
import boto3
import pickle
# import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime


def clean_gps(df):
    """
    Renombrar las columnas y sacar espacios vacios en la data de gps

    Parameters
    ----------
    df : dataframe
        gps data.

    Returns
    -------
    df : dataframe
        gps data renombrada y ordenada.


    """
    columnas = ['Equipo', 'Date', 'Velocidad', 'Norte', 'Este', 'Cota']
    df = df[columnas]
    df.rename(columns={"Equipo": "equipo", "Date": "date",
                       "Velocidad": "velocidad", "Este": "este",
                       "Norte": "norte", "Cota": "cota"},
              inplace=True)
    # forma mas rápida de filtrar el dataset completo
    df["prefijo"] = df["equipo"].apply(lambda x: x[0:3])
    df = df[df["prefijo"] == "CDH"]
    df.drop(columns=["prefijo"], inplace=True)
    # eliminar los espacios en las columnas
    # df["equipo"] = df[df["equipo"].str.contains("CDH")]
    # eliminar los espacios en las columnas
    df = drop_spaces_data(df)
    # transformar a datetime
    # df["date"] = df["date"].apply(lambda x: transform_date(x))
    df.drop_duplicates(subset=["equipo", "date"], inplace=True)
    # eliminar los que no traen datos de equipos
    df.dropna(subset=["equipo"], inplace=True)
    # reset indexes
    df.reset_index(drop=True, inplace=True)
    return df


def clean_loads(df):
    """
    Renombrar las columnas y sacar espacios vacios en la data de laods

    Parameters
    ----------
    df : dataframe
        laods data.

    Returns
    -------
    df : dataframe
        loads data renombrada y ordenada.

    """
    columnas = ['IdTurno', 'Equipo', 'Date', 'Origen', 'UbicacionDescarga',
                'Date descarga inicio', 'Date descarga fin', 'Toneladas',
                'Operador', 'Grupo']
    df = df[columnas]
    df.rename(columns={"IdTurno": "id_turno", "Equipo": "equipo",
                       "Date": "date", "Origen": "origen",
                       "UbicacionDescarga": "destino",
                       "Date descarga inicio": "fecha_inicio_descarga",
                       "Date descarga fin": "fecha_fin_descarga",
                       "Toneladas": "tonelaje",
                       "Operador": "operador",
                       "Grupo": "grupo"},
              inplace=True)
    # eliminar los espacios en las columnas
    df = drop_spaces_data(df)
    df["operador"] = df["operador"].str.lower().str.strip()
    df["operador"] = df["operador"].fillna("nr")
    # reemplazar grupos nombre
    df["grupo"] = df["grupo"].str.replace("Grupo ", "G").replace(" ", "")
    # eliminar aquellos que no han descargado aún
    df.dropna(subset=["fecha_inicio_descarga"], inplace=True)
    df.dropna(subset=["fecha_fin_descarga"], inplace=True)
    df["tiempo_viaje"] = (
        df["fecha_fin_descarga"] - df["date"]).dt.total_seconds() / 60
    # dejar bien ordenados los origenes y destinos
    df["origen"] = df["origen"].str.replace("/", "-")
    df["destino"] = df["destino"].str.replace("/", "-")
    df["origen"] = df["origen"].str.replace(" ", "-")
    df["destino"] = df["destino"].str.replace(" ", "-")
    # corregir cifras de traslado
    df["tonelaje"] = df["tonelaje"].\
        apply(lambda x: 300 if x > 400 else x)
    df["tonelaje"] = df["tonelaje"].\
        apply(lambda x: 300 if x < 100 else x)
    # transformar las fechas

    # df["date"] = df["date"].apply(lambda x: transform_date(x))
    # # transformar fecha_inicio_descarga
    # df["fecha_inicio_descarga"] =\
    #     df["fecha_inicio_descarga"].apply(lambda x: transform_date(x))
    # # transformar fecha_fin_descarga
    # df["fecha_fin_descarga"] =\
    #     df["fecha_fin_descarga"].apply(lambda x: transform_date(x))

    # eliminar los duplicados
    df.drop_duplicates(subset=["equipo", "date"], inplace=True)
    # borrar los que no tienen equipo
    df.dropna(subset=["equipo"], inplace=True)

    # reset indexes
    df.reset_index(drop=True, inplace=True)
    return df


def timezone_fechas(zona, fecha):
    """
    La funcion entrega el formato de zona horaria a las fechas de los
    dataframe
    Parameters
    ----------
    zona: zona horaria a usar
    fecha: fecha a modificar
    Returns
    -------
    fecha_zh: fecha con el fomarto de zona horaria
    """
    # Definimos la zona horaria
    timezone = pytz.timezone(zona)
    fecha_zs = timezone.localize(fecha)

    return fecha_zs


def drop_spaces_data(df):
    """
    sacar los espacios de columnas que podrián venir interferidas
    Parameters
    ----------
    df : dataframe
        input data
    column : string
        string sin espacios en sus columnas
    Returns
    -------
    """
    for column in df.columns:
        try:
            df[column] = df[column].str.lstrip()
            df[column] = df[column].str.rstrip()
        except Exception as e:
            print(e)
            pass
    return df


def resample_gps(gps, freq="5S"):
    """
    Hacer resample de los puntos de gps a freq segundos de proximidad
    Parameters
    ----------
    gps : dataframe
        dataframe de gps.
    freq : string, optional
        frecuencia de resampleo. The default is "5S".
    Returns
    -------
    salida : dataframe
        dataframe de gps resampleado.
    """
    salida = pd.DataFrame()
    # resample de gps
    for equipo in gps["equipo"].unique():
        print(equipo)
        gps_equipo = gps[gps["equipo"] == equipo]
        gps_equipo.sort_values(by=["fecha"], inplace=True)
        gps_equipo.reset_index(inplace=True, drop=True)
        gps_equipo.set_index(["fecha"], inplace=True)
        gps_equipo = gps_equipo.resample(freq).asfreq()
        # relleno de vacios
        gps_equipo["equipo"].fillna(value=equipo, inplace=True)
        gps_equipo["velocidad"].fillna(method="bfill", inplace=True)

        gps_equipo = gps_equipo.interpolate(
            method='linear', limit_direction='backward', axis=0)
        gps_equipo.reset_index(drop=False, inplace=True)
        salida = pd.concat([salida, gps_equipo], axis=0)
    salida.sort_values(by=["equipo", "fecha"], inplace=True)
    salida.reset_index(drop=True, inplace=True)
    return salida


def transform_date(date):
    """
    Transformar la fecha y formato de timestamp según el formato que venian
    de jigsaw
    Parameters
    ----------
    date : TYPE
        DESCRIPTION.
    Returns
    -------
    fecha_loads : TYPE
        DESCRIPTION.
    """
    date = str(date)
    fecha = date[0:10]
    hora = date[11:19]
    date = fecha + " " + hora
    fecha_loads = datetime.strptime(date,
                                    "%Y-%m-%d %H:%M:%S")
    fecha_loads = timezone_fechas("America/Santiago", fecha_loads)
    return fecha_loads


def convert_df_float(df):
    """
    Convertir columnas de un dataframe en flotantes

    Parameters
    ----------
    df : dataframe
        dataframe a pasr a flotante.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    for col in df.columns:
        try:
            df[col] = df[col].apply(float)
        except Exception:
            pass
    df.reset_index(drop=True, inplace=True)
    return df


def read_pkl_s3(bucket, ruta):
    """
    La funcion lee un archivo pkl desde s3
    Parameters
    ----------
    bucket : Nombre del bucket
    ruta : Ruta del archivo
    Returns
    -------
    data : Dataframe con los datos
    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket,
                        Key=ruta)
    body = obj["Body"].read()
    data = pickle.loads(body)
    data.reset_index(inplace=True, drop=True)
    return data


def read_csv_s3(s3_bucket, s3_event_key):
    """
    Leer archivo .csv desde s3
    Parameters
    ----------
    bucket : string
        nombre del bucket.
    ruta : string
        ruta de s3 ddonde ir a buscar la data.\
    Returns
    -------
    data : dataframe
        lectura del archivo como un pandas dataframe
    Alternativa:
    obj = s3.get_object(Bucket=s3_bucket,
                        Key=s3_event_key)
    body = io.BytesIO(obj["Body"].read())
    # leer la data
    data = pd.read_csv(body, sep=",",
                        index_col=0,
                        encoding="ISO-8859-1")
    """
    s3 = boto3.client("s3")
    csv_obj = s3.get_object(Bucket=s3_bucket, Key=s3_event_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(StringIO(csv_string), index_col=0)
    return data
