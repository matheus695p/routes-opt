import heapq
import collections
import numpy as np


def shortest_path(edges, source, sink):
    """
    Ruta mas corta según el camino de puntos

    Parameters
    ----------
    edges : list
        Lista de los caminos posibles, ejemplo:
            [("pti", "ptj", "dij"), .... , ("pti_n", "ptj_n", "dij_n")].
    source : string
        punto de inicio.
    sink : string
        donde se quiere llegar.
    Returns
    -------
    list
        camino más corto que se debe recorrer.
    """
    # crear pesos DAG - {node:[(cost,neighbour), ...]}
    graph = collections.defaultdict(list)
    for l, r, c in edges:
        graph[l].append((c, r))
    # crear una cola de prioridad y un conjunto de hash para almacenar los
    # nodos visitados
    queue, visited = [(0, source, [])], set()
    heapq.heapify(queue)
    # gráfico transversal con BFS
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        # visitar el nodo si no fue visitado antes
        if node not in visited:
            visited.add(node)
            path = path + [node]
            # golpea el fregadero
            if node == sink:
                return (cost, path)
            # visitar vecindades
            for c, neighbour in graph[node]:
                if neighbour not in visited:
                    heapq.heappush(queue, (cost+c, neighbour, path))
    return np.nan


def track(df, array_punto):
    """
    Selección de camino
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    array_punto : TYPE
        DESCRIPTION.
    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    punto = array_punto[0]
    puntos = array_punto[1]
    track = shortestPath(puntos, df["truck_point"], df[punto])
    try:
        return track[0]
    except TypeError:
        return np.nan


def create_list_points(dataframe, points_list):
    """
    Crear lista de puntos
    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.
    points_list : TYPE
        DESCRIPTION.
    Returns
    -------
    None.
    """
    points_list.append((dataframe["origen"], dataframe["destino"],
                        dataframe["distancia"]))
    points_list.append((dataframe["destino"], dataframe["origen"],
                        dataframe["distancia"]))
