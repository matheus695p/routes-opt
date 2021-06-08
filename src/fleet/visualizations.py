import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.fleet.utils import try_create_folder
plt.style.use('dark_background')
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)


def plot_time_series(df, fecha_inicial="2018-01-01 00:00:00",
                     fecha_final="2020-03-10 05:30:00",
                     title="Evolución de variables flotación",
                     ylabel="None",
                     sample=9):
    """
    Plot de la serie de tiempo que tiene como indice la fehca en formato
    timestamp
    Parameters
    ----------
    df : dataframe
        datos de las series de tiempo ordenados hacia el lado.
    fecha_inicial : str
        fecha inicial en el formato %y-%m-%d.
        The default is "2018-01-01 00:00:00".
    fecha_final : TYPE
        fecha final en el formato %y-%m-%d.
        The default is "2020-03-10 05:30:00".
    title : str, optional
        titulo. The default is "Evolución de variables flotación".
    ylabel : TYPE, optional
        nombre de eje Y. The default is "None".
    sample : TYPE, optional
        cuantas columans quieres tomar. The default is 9.
    Returns
    -------
    Gráficos en una sola plana
    """
    # fig, axs = plt.subplots(3, 3, figsize=(40, 40), sharex=True)
    # axx = axs.ravel()
    max_counter = int(len(df.columns) / sample) + 1
    columns = list(df.columns)

    for j in range(0, max_counter):
        fig, axs = plt.subplots(3, 3, figsize=(40, 40), sharex=True)
        axx = axs.ravel()

        for i in range(j*sample, (j+1)*sample):
            if i >= len(columns):
                pass
            else:
                alpha = i - j*sample
                df[columns[i]].loc[fecha_inicial:fecha_final].plot(
                    ax=axx[alpha])
                axx[alpha].set_xlabel("Fecha [dias]", fontsize=40)
                axx[alpha].set_ylabel(columns[i], fontsize=40)
                axx[alpha].set_title(title, fontsize=30)
                axx[alpha].grid(which='minor', axis='x')
                name = str(j)
        fig.savefig(f"images/{name}.png")


def plot_instance_training(history, epocas_hacia_atras, model_name,
                           filename):
    """
    Sacar el historial de entrenamiento de epocas en partivular
    Parameters
    ----------
    history : object
        DESCRIPTION.
    epocas_hacia_atras : int
        epocas hacia atrás que queremos ver en el entrenamiento.
    model_name : string
        nombre del modelo.
    filename : string
        nombre del archivo.
    Returns
    -------
    bool
        gráficas de lo ocurrido durante el entrenamiento.
    """
    plt.style.use('dark_background')
    letter_size = 20
    # Hist training
    largo = len(history.history['loss'])
    x_labels = np.arange(largo-epocas_hacia_atras, largo)
    x_labels = list(x_labels)
    # Funciones de costo
    loss_training = history.history['loss'][-epocas_hacia_atras:]
    loss_validation = history.history['val_loss'][-epocas_hacia_atras:]
    # Figura
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax.plot(x_labels, loss_training, 'gold', linewidth=2)
    ax.plot(x_labels, loss_validation, 'r', linewidth=2)
    ax.set_xlabel('Epocas', fontname="Arial", fontsize=letter_size-5)
    ax.set_ylabel('Función de costos', fontname="Arial",
                  fontsize=letter_size-5)
    ax.set_title(f"{model_name}", fontname="Arial", fontsize=letter_size)
    ax.legend(['Entrenamiento', 'Validación'], loc='upper left',
              prop={'size': letter_size-5})
    # Tamaño de los ejes
    for tick in ax.get_xticklabels():
        tick.set_fontsize(letter_size-5)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(letter_size-5)
    plt.show()
    return fig


def training_history(history, model_name="NN", filename="NN"):
    """
    Según el historial de entrenamiento que hubo plotear el historial
    hacía atrás de las variables
    Parameters
    ----------
    history : list
        lista con errores de validación y training.
    model_name : string, optional
        nombre del modelo. The default is "Celdas LSTM".
    filename : string, optional
        nombre del archivo. The default is "LSTM".
    Returns
    -------
    None.
    """
    size_training = len(history.history['val_loss'])
    fig = plot_instance_training(history, size_training, model_name,
                                 filename + "_ultimas:" +
                                 str(size_training) + "epocas")

    fig = plot_instance_training(history, int(1.5 * size_training / 2),
                                 model_name,
                                 filename + "_ultimas:" +
                                 str(1.5 * size_training / 2) + "epocas")
    # guardar el resultado de entrenamiento de la lstm
    print(os.getcwd())
    try_create_folder("results")
    fig.savefig(f"results/{model_name}_training.png")

    fig = plot_instance_training(history, int(size_training / 2),
                                 model_name,
                                 filename + "_ultimas:" + str(
                                     size_training / 2) + "epocas")

    fig = plot_instance_training(history, int(size_training / 3), model_name,
                                 filename + "_ultimas:" +
                                 str(size_training / 3) + "epocas")
    fig = plot_instance_training(history, int(size_training / 4), model_name,
                                 filename + "_ultimas:" + str(
                                     size_training / 4) + "epocas")
    print(fig)


def plot_xy_results(predictions, real, index=str(1), name="col",
                    folder_name="nn"):
    """
    Plot sequence de la secuecnia
    Parameters
    ----------
    predictions : array
        predicciones.
    real : array
        valores reales.
    fechas : array
        array de fechas.
    indice : TYPE
        indice de la columna.
    Returns
    -------
    plot de prediciones vs real.
    """
    plt.style.use('dark_background')
    letter_size = 20
    mae = np.abs(predictions - real).mean()
    percentage = np.abs(predictions - real) / real * 100
    mean_percentage = str(round(percentage.mean(), 2))
    kernel_name = name + \
        " Distribución [%]" + "\n" + f"MAPE: {mean_percentage} [%]"
    # plot del histograma de errores
    kernel_density_estimation(pd.DataFrame(percentage,
                                           columns=["error"]), "error",
                              name=kernel_name, bw_adjust=0.5)
    kernel_density_estimation(pd.DataFrame(np.abs(predictions - real),
                                           columns=["error"]), "error",
                              name=kernel_name, bw_adjust=0.5)
    mae = round(mae, 3)
    # caracteristicas del dataset
    mean = round(real.mean(), 3)
    mean = round(mean, 3)

    std = round(real.std(), 3)

    fig, ax = plt.subplots(1, figsize=(22, 12))
    plt.scatter(real, predictions, color='orangered')
    plt.scatter(real, real, color='green')
    titulo = f"Predicciones {name} --> error: {str(mae)}" + "\n" +\
        f"Caracteristicas: promedio: {str(mean)}, desv: {str(std)}"
    plt.title(titulo, fontsize=30)
    plt.xlabel('Real', fontsize=30)
    plt.ylabel(f'Predicción {folder_name}', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(['real', 'predicción'], loc='upper left',
               prop={'size': letter_size+5})
    # plt.ylim(0, 4600)
    # plt.xlim(0, 4600)
    plt.show()

    path = f"results/{folder_name}/{index}-{name}.png"
    fig.savefig(path)


def plot_multiple_xy_results(predictions, y_test, target_cols, ind,
                             folder_name="nn"):
    """
    Plotea lo resutltados de la red, cuando es con más de un input
    Parameters
    ----------
    predictions : array
        predicciones.
    real : array
        valores reales.
    names : list
        nombre de las columans target.
    folder_name : string, optional
        directorio donde se dejaran las carpetas. The default is "nn".
    Returns
    -------
    Plot.
    """
    try_create_folder("results")
    try_create_folder(f"results/{folder_name}")
    for i in range(predictions.shape[1]):
        print("Resultados:", target_cols[i], "...")
        predi = predictions[:, i]
        reali = y_test[:, i]
        namei = target_cols[i]
        plot_xy_results(predi, reali, index=ind, name=namei,
                        folder_name=folder_name)


def kernel_density_estimation(df, col, name="Entrenamiento", bw_adjust=0.1):
    """
    Estimación de la densidad a través de un kernel
    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    col : string
        nombre de la columna a realizar el violinplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".
    bw_adjust : float, optional
        Ajuste de la distribución. The default is 0.1.
    Returns
    -------
    Estimación de la distribución de la columna.
    """
    sns.set(font_scale=1.5)
    plt.style.use('dark_background')
    # pplot = sns.displot(df, x=col, kind="kde", bw_adjust=bw_adjust)
    pplot = sns.displot(df, x=col, kind="kde")
    pplot.fig.set_figwidth(10)
    pplot.fig.set_figheight(8)
    pplot.set(title=name)


def watch_distributiions(y_train, y_test, target_cols):
    """
    Ver las distribuciones de los targets
    Parameters
    ----------
    y_train : numpy array
        DESCRIPTION.
    y_val : numpy array
        DESCRIPTION.
    y_test : numpy array
        DESCRIPTION.
    Returns
    -------
    None.
    """
    for i in range(y_train.shape[1]):

        kernel_density_estimation(pd.DataFrame(
            y_train[:, i], columns=["y_train"]), "y_train",
            name="Entrenamiento"+target_cols[i])
        # testing
        kernel_density_estimation(pd.DataFrame(
            y_test[:, i], columns=["y_test"]), "y_test",
            name="Testing"+target_cols[i])
