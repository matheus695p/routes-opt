# import pandas as pd


def nn_preparation(df, target_col, timesteps=5):
    """
    Hacer la preparación de la red neuronal
    Parameters
    ----------
    df : dataframe
        array con todas las variables.
    target_col : string or list
        nombre de la/as columna/as target/s.

    Returns
    -------
    x : array
        x en numpy.
    y : array
        target en numpy.
    """
    df.reset_index(drop=True, inplace=True)
    x = df.drop(columns=target_col)
    y = df[target_col]
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y


def inter_quantile_range(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    # low_lim = q1 - 1.5 * iqr
    low_lim = q1 - 1.5 * iqr
    up_lim = q3 + 1.5 * iqr
    df = df[(df[column] > low_lim) & (df[column] < up_lim)]
    df.reset_index(drop=True, inplace=True)
    return df
