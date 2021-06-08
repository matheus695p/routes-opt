import pandas as pd
import numpy as np
import xgboost as xgb
from copy import copy


def create_guess(points):
    """
    Crea una ruta posible entre todos los puntos, volviendo al original.
    Entrada: Lista de ID de puntos

    Parameters
    ----------
    points: array, list
        puntos por los que tiene que pasar con ids.

    Returns
    -------
    TYPE
        rutas posible random.

    """

    guess = copy(points)
    np.random.shuffle(guess)
    guess.append(guess[0])
    return list(guess)


def create_generation(points, population=100):
    """
    Hace una lista de órdenes de puntos adivinados dada una lista de ID de
    puntos.

    Parameters
    ----------
    points : list
        lista de ids de puntos.
    population : int, optional
        cuántas rutas hacer. The default is 100.

    Returns
    -------
    generation : TYPE
        DESCRIPTION.

    """
    generation = [create_guess(points) for _ in range(population)]
    return generation


def travel_time_between_points(loaded_model, my_date,
                               point1_id, point2_id,
                               hour, date, passenger_count=1,
                               store_and_fwd_flag=0, pickup_minute=0):
    """
    Dados dos puntos, esto calcula el viaje entre ellos basándose en un XGBoost
    modelo predictivo, esto también se puede hacer con la red neuronal

    """

    model_data = {'passenger_count': passenger_count,
                  'pickup_longitude': point1_id[1],
                  'pickup_latitude': point1_id[0],
                  'dropoff_longitude': point2_id[1],
                  'dropoff_latitude': point2_id[0],
                  'store_and_fwd_flag': store_and_fwd_flag,
                  'pickup_month': my_date.month,
                  'pickup_day': my_date.day,
                  'pickup_weekday': my_date.weekday(),
                  'pickup_hour': hour,
                  'pickup_minute': pickup_minute,
                  'latitude_difference': point2_id[0] - point1_id[0],
                  'longitude_difference': point2_id[1] - point1_id[1],
                  'trip_distance': calculate_distance(point1_id, point2_id)
                  }
    # convertir a dataframe
    df = pd.DataFrame([model_data], columns=model_data.keys())
    # hacer las predicciones del tiempo que se demoraria
    pred = np.exp(loaded_model.predict(xgb.DMatrix(df))) - 1

    return pred[0]


def calculate_distance(point1_id, point2_id):
    """
    Calcular la distancia entre dos puntos a través de su información
    georeferencial

    Parameters
    ----------
    point1_id : array, list
        punto 1.
    point2_id : array, list
        punto 2.

    Returns
    -------
    dist : float
        Distancia entre los dos puntos.

    """
    dist = 0.621371 *\
        6371 * (abs(2 * np.arctan2(
            np.sqrt(
                np.square(np.sin(
                    (abs(point2_id[0] - point1_id[0]) * np.pi / 180) / 2))),
            np.sqrt(1-(
                np.square(np.sin(
                    (abs(
                        point2_id[0] - point1_id[0]) * np.pi / 180) / 2)))))) +
                abs(2 * np.arctan2(
                    np.sqrt(
                        np.square(
                            np.sin((abs(point2_id[1] -
                                        point1_id[1]) * np.pi / 180) / 2))),
                    np.sqrt(1-(
                        np.square(
                            np.sin((abs(point2_id[1] -
                                        point1_id[1]) * np.pi / 180) / 2)))))))
    return dist


def fitness_score(guess, loaded_model, coordinates, my_date):
    """
    Recorre los puntos en el orden de guesses y calcula
    cuánta distancia tomaría el camino para completar un bucle.
    Más bajo es mejor.

    Parameters
    ----------
    guess : list
        lista de puntos.
    loaded_model : model
        modelo que calcula las distancias.
    coordinates : dict
        coordenadas a pasar.
    my_date : datetime
        tiempo que se quiere evaluar.

    Returns
    -------
    score : float
        cuanto tiempo tomaria el camino escogido.

    """

    score = 0
    for ix, point_id in enumerate(guess[:-1]):
        score += travel_time_between_points(
            loaded_model, my_date, coordinates[point_id],
            coordinates[guess[ix+1]], 11, my_date)
    return score


def check_fitness(guesses, loaded_model, coordinates, my_date):
    """
    Pasa por todas las rutas (guesses) y calcula fitness_score.
    Devuelve una lista de tuplas: (guess, fitness_score)

    Parameters
    ----------
    guess : list
        lista de puntos.
    loaded_model : model
        modelo que calcula las distancias.
    coordinates : dict
        coordenadas a pasar.
    my_date : datetime
        tiempo que se quiere evaluar.

    Returns
    -------
    fitness_indicator : TYPE
        DESCRIPTION.

    """
    fitness_indicator = []
    for guess in guesses:
        fitness_indicator.append((guess, fitness_score(
            guess, loaded_model, coordinates, my_date)))
    return fitness_indicator


def get_breeders_from_generation(guesses, loaded_model, coordinates,
                                 my_date, take_best_N=10,
                                 take_random_N=5, verbose=False,
                                 mutation_rate=0.1):
    """
    Esto establece el grupo de cría para la próxima generación. Tu tienes
    tener mucho cuidado con cuántos criadores tomas, de lo contrario tu
    la población puede explotar. Estos dos, más el "número de niños por
    pareja "en la función make_children debe ajustarse para evitar exponencial
    crecimiento o declive!

    Parameters
    ----------
    guess : list
        lista de puntos.
    loaded_model : model
        modelo que calcula las distancias.
    coordinates : dict
        coordenadas a pasar.
    my_date : datetime
        tiempo que se quiere evaluar.
    take_best_N : int, optional
        Generacion que cria para la siguente, los mejores. The default is 10.
    take_random_N : TYPE, optional
        Tomar aleatorio para que haya mezcla. The default is 5.
    verbose : boolean, optional
        imprimir o no. The default is False.
    mutation_rate : float, optional
        número entre 0-1 para tasa de mutación. The default is 0.1.

    Returns
    -------
    new_generation : list
        siguente generación.
    best_guess : list
        ruta optima, la que esta dando mejor hasta el momento.

    """

    # Primero, obtener las mejores suposiciones de la última vez
    fit_scores = check_fitness(guesses, loaded_model, coordinates,
                               my_date)
    #  ordenar el bajo es primero, que es el que queremos
    sorted_guesses = sorted(fit_scores, key=lambda x: x[1])
    new_generation = [x[0] for x in sorted_guesses[:take_best_N]]
    best_guess = new_generation[0]

    if verbose:
        # ¡Si queremos ver cuál es la mejor suposición actual!
        print(best_guess)

    #  En segundo lugar, obtenga algunos aleatorios para la
    # diversidad genética.
    for _ in range(take_random_N):
        ix = np.random.randint(len(guesses))
        new_generation.append(guesses[ix])

    # No hay mutaciones aquí ya que el orden realmente importa.
    # Si quisiéramos, podríamos agregar una mutación de "intercambio",
    # pero en la práctica no parece ser necesario

    np.random.shuffle(new_generation)
    return new_generation, best_guess


def make_child(parent1, parent2):
    """
    Tome algunos valores del padre 1 y manténgalos en su lugar,
    luego fusione los valores de parent2, completando de izquierda a
    derecha con ciudades que no
    ya en el niño.

    Parameters
    ----------
    parent1 : list.
        Ruta 1
    parent2 : list.
        Ruta 2

    Returns
    -------
    child : list
        ruta creada.

    """

    list_of_ids_for_parent1 = list(np.random.choice(
        parent1, replace=False, size=len(parent1)//2))
    child = [-99 for _ in parent1]

    for ix in range(0, len(list_of_ids_for_parent1)):
        child[ix] = parent1[ix]
    for ix, gene in enumerate(child):
        if gene == -99:
            for gene2 in parent2:
                if gene2 not in child:
                    child[ix] = gene2
                    break
    child[-1] = child[0]
    return child


def make_children(old_generation, children_per_couple=1):
    """
    Empareja a los padres y crea hijos para cada pareja.
    Si hay un número impar de posibilidades de padres, uno
    quedará fuera, muy sad xd.

    El emparejamiento se produce al emparejar la primera y la última entrada.
    Luego el segundo y el segundo desde el último, y así sucesivamente.

    Parameters
    ----------
    old_generation : list
        rutas de la antigua generación.
    children_per_couple : int, optional
        numero de rutas posibles por pareja. The default is 1.

    Returns
    -------
    next_generation : list
        proxima generación.

    """

    mid_point = len(old_generation)//2
    next_generation = []

    for ix, parent in enumerate(old_generation[:mid_point]):
        for _ in range(children_per_couple):
            next_generation.append(make_child(parent, old_generation[-ix-1]))
    return next_generation


def evolve_to_solve(current_generation, loaded_model, coordinates, my_date,
                    max_generations, take_best_N,
                    take_random_N, mutation_rate, children_per_couple,
                    print_every_n_generations, verbose=False):
    """
    Toma una generación de guesses y luego las evoluciona con el tiempo
    utilizando nuestro reglas de crianza.
    Continúe con esto durante "max_generations" veces.

    Parameters
    ----------
    current_generation : list
        la primera generación de conjeturas.
    loaded_model : model
        modelo que calcula las distancias.
    coordinates : dict
        coordenadas a pasar.
    my_date : datetime
        tiempo que se quiere evaluar.
    max_generations : int
        cuántas generaciones completar.
    take_best_N : list
        cuántos de los mejores resultados se seleccionan para criar.
    take_random_N : list
        cuántas suposiciones aleatorias se introducen para mantener
        la genética.
    mutation_rate : float
        Con qué frecuencia mutar (actualmente sin usar).
    children_per_couple : list
        Cuántos hijos por pareja reproductora.
    print_every_n_generations : int
        frecuencia que se muestra.
    verbose : boolean, optional
        muestra impresiones de progreso. The default is False.

    Returns
    -------
    fitness_tracking : list
        Una lista del puntaje de condición física en cada generación.
    best_guess : list
        el best_guess al final de la evolución.

    """
    fitness_tracking = []
    for i in range(max_generations):
        if verbose and not i % print_every_n_generations and i > 0:
            print("Generacion %i: " % i, end='')
            print(len(current_generation))
            print("Mejor Score Actual: ", fitness_tracking[-1])
            is_verbose = True
        else:
            is_verbose = False
        breeders, best_guess =\
            get_breeders_from_generation(current_generation, loaded_model,
                                         coordinates, my_date,
                                         take_best_N=take_best_N,
                                         take_random_N=take_random_N,
                                         verbose=is_verbose,
                                         mutation_rate=mutation_rate)
        fitness_tracking.append(fitness_score(best_guess, loaded_model,
                                              coordinates, my_date))
        current_generation = make_children(
            breeders, children_per_couple=children_per_couple)

    return fitness_tracking, best_guess
