import pandas as pd
from src.utils.visualizations import kernel_density_estimation

distribution = pd.read_pickle("data/raw-data/gps/angle_distribution.pkl")
distribution = distribution[(distribution["angulo"] > -60)
                            & (distribution["angulo"] < 60)]
distribution.reset_index(drop=True, inplace=True)


for trayecto in distribution["trayecto"].unique():
    print(trayecto)
    df_ii = distribution[distribution["trayecto"] == trayecto]
    df_ii.reset_index(drop=True, inplace=True)
    kernel_density_estimation(df_ii, "angulo", name="trayecto", bw_adjust=1)
