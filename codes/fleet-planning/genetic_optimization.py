import pickle
import datetime
from geopy.geocoders import Nominatim
from src.genetic.genetic_module import (create_guess, create_generation,
                                        check_fitness,
                                        get_breeders_from_generation,
                                        make_children, evolve_to_solve)
# cargar el modelo preentrenado
filename = "models/xgboost.sav"
loaded_model = pickle.load(open(filename, 'rb'))

# 27 de abril del 2021 [como es data antigua mejor escoger un fecha antigua]
date_list = [27, 4, 2016]
year = int(date_list[2])
month = int(date_list[1])
day = int(date_list[0])

# fecha de prueba
my_date = datetime.date(year, month, day)

# lugares de prueba [poner las ubicaciones que debe realizar]
test_locations = {'L1': (40.819688, -73.915091),
                  'L2': (40.815421, -73.941761),
                  'L3': (40.764198, -73.910785),
                  'L4': (40.768790, -73.953285),
                  'L5': (40.734851, -73.952950),
                  'L6': (40.743613, -73.977998),
                  'L7': (40.745313, -73.993793),
                  'L8': (40.662713, -73.946101),
                  'L9': (40.703761, -73.886496),
                  'L10': (40.713620, -73.943076),
                  'L11': (40.725212, -73.809179)}

# localizarlos en un mapa
geolocator = Nominatim(user_agent="hola_sapo_qlo")
addresses = []

# descomprimir los lugares donde se esta haciendo esto y dejarlos en strings
for key in test_locations:
    location = geolocator.reverse(test_locations[key])
    addresses.append(location.address)

# crear primer genoma
create_guess(list(test_locations.keys()))
test_generation = create_generation(list(test_locations.keys()),
                                    population=10)
print(test_generation)

# coordenadas a suplir
coordinates = test_locations.copy()
print(check_fitness(test_generation, loaded_model, coordinates, my_date))

# crear la generacion inicial
current_generation = create_generation(
    list(test_locations.keys()), population=500)

# hacer la inferencia para esos caminos
fitness_tracking, best_guess = evolve_to_solve(
    current_generation, loaded_model, coordinates, my_date, 100, 150,
    70, 0.5, 3, 5, verbose=True)

print("El camino que minimiza los tiempos de viaje es:",
      best_guess)
