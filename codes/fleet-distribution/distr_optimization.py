import random
import warnings
import pandas as pd
from pyomo.environ import (ConcreteModel, Var, NonNegativeReals,
                           Objective, minimize, ConstraintList,
                           SolverFactory, Constraint)
from src.ads.cleaning import (clean_loads)
from src.optimization.opt_module import (lower_limit_constraint,
                                         upper_limit_constraint)
warnings.filterwarnings("ignore")

# esta mierda hay que poner el path a donde esta instalado el solver
solvername = 'glpk'

solverpath_exe =\
    r"C:\Users\mateu\anaconda3\envs\routes-optimization\Library\bin\glpsol"
# reproductibilidad
seed = 21
random.seed(seed)

# traerme la data de loads
loads = pd.read_pickle("data/raw-data/loads/loads-2020-07-15.pkl")
loads = clean_loads(loads)

# agregar leyes de cobre aleatorias
copper_law = []
for origen in loads["origen"].unique():
    copper_law.append([origen, random.gauss(0.15, 0.05)])
copper_law = pd.DataFrame(copper_law, columns=["origen", "ley_cobre"])
copper_law["ley_cobre"] = copper_law["ley_cobre"].apply(
    lambda x: 0.1 + random.gauss(0.1, 0.0000005) if x < 0 else x)

# concatenar los puntos con las leyes de cobre
loads = loads.merge(copper_law, on=["origen"], how="outer")
# crear el destino y el tiempo medio de viaje entre origen y destino
loads["trayecto"] = loads["origen"] + " / " + loads["destino"]
# establecer la función a optimizar
average_matrix = loads.groupby("trayecto").mean()[["tiempo_viaje",
                                                   "tonelaje"]]
average_matrix.reset_index(drop=False, inplace=True)
# matriz a optimizar
opt_matrix = average_matrix.merge(
    loads[["ley_cobre", "trayecto"]], on=["trayecto"], how="inner")
opt_matrix.drop_duplicates(inplace=True)
opt_matrix.reset_index(drop=True, inplace=True)

# meta productiva [por dia aproximada]
productive_goal = 160000
# tamaño de la flota
fleet_size = loads["equipo"].nunique()
# cantidad de minutos disponibles para realizarlo
total_hours = 24 * 60 * fleet_size

# definir los parametros de la función de optimización
avg_time_traveled = opt_matrix["tiempo_viaje"].to_list()
avg_tons_transported = opt_matrix["tonelaje"].to_list()
avg_copper_laws = opt_matrix["tonelaje"].to_list()
# definir nombres
routes = opt_matrix["trayecto"].to_list()
ids = range(len(opt_matrix))
# empezar la optimización [modelo]
model = ConcreteModel()

# variables
model.routes = Var(routes, domain=NonNegativeReals)

# función objectivo que es el tiempo total
total_time = sum(avg_time_traveled[i] * model.routes[routes[i]]
                 for i in range(len(routes)))
model.objective = Objective(expr=total_time, sense=minimize)

# restricciones [se debe llegar a la meta productiva]
model.constraints = ConstraintList()

# for p in ids:
#     model.constraints.add(10 <= model.routes[routes[p]] <= 200)
for p in ids:
    model.constraints.add(0 <= model.routes[routes[p]] <= 200)

# se debe cumplir la meta productiva
model.constraints.add(
    expr=sum(model.routes[routes[i]] *
             avg_copper_laws[i] *
             avg_tons_transported[
                 i] for i in range(len(routes))) >= productive_goal)

# se debe cumplir la meta productiva
model.constraints.add(
    expr=sum(model.routes[
        routes[i]] * avg_time_traveled[i] for i in ids) <= total_hours)

# model.objective.pprint()
model.constraints.pprint()

# resolver la ecuación
solver = SolverFactory(solvername, executable=solverpath_exe)
results = solver.solve(model, tee=True)
# model.pprint()
# print(solver.solve(model))

results.write()
