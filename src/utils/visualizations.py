import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from mpl_toolkits.mplot3d import axes3d
warnings.filterwarnings("ignore")
plt.style.use('dark_background')


def vista_eje_z(df1, df2, filename="path"):
    # puntos intrinsecos del mapa gps
    x1, y1, z1 =\
        df1[["norte"]].to_numpy(), df1[["este"]].to_numpy(), df1[[
            "cota"]].to_numpy()
    # puntos a graficar del mapa
    x2, y2, z2 =\
        df2[["norte"]].to_numpy(), df2[["este"]].to_numpy(), df2[[
            "cota"]].to_numpy()
    # # Creamos la figura
    fig = plt.figure(figsize=(20, 15))
    # Creamos el plano 3D
    ax = fig.gca(projection='3d')
    # # Agregamos los puntos en el plano 3D
    ax.set_xlabel('Este:  Coordenada X', fontsize=18)
    ax.set_ylabel('Norte:  Coordenada Y', fontsize=18)
    ax.set_zlabel('Elevación: Coordenada Z', fontsize=18)
    ax.set_title('Minera XXX - Grupo xxx', fontsize=24)
    # ax.scatter(x1, y1, z1, c=(0.6, 0.3, 0.1), marker='*')
    # ax.scatter(x1, y1, z1, s=np.pi*0.003*100, c=(0.6, 0.3, 0.1), alpha=1)
    ax.scatter(x1, y1, z1, s=np.pi*0.003*100, c="darkorange", alpha=1)
    ax.scatter(x2, y2, z2, s=np.pi*0.3*100, c="k", alpha=1)
    ax.scatter(x2[0], y2[0], z2[0], s=np.pi*0.3 *
               100*20, c="midnightblue", alpha=1)
    ax.scatter(x2[-1], y2[-1], z2[-1], s=np.pi *
               0.3*100*20, c="midnightblue", alpha=1)

    # limites del plot
    ax.set_xlim3d(97000, 102000)
    ax.set_ylim3d(97000, 102000)
    # Vista Eje Z
    # ax.view_init(75, 15)
    ax.view_init(90, 0)
    plt.show()
    fig.savefig(filename)


def kernel_density_estimation(df, col, name="Colum", bw_adjust=0.1):
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
    pplot = sns.displot(df, x=col, kind="kde", bw_adjust=bw_adjust)
    pplot.fig.set_figwidth(10)
    pplot.fig.set_figheight(8)
    pplot.set(title=name)
