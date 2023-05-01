import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time

from aux_functions import *
from generation import generation as gen
from local_search import local_search as ls

n, quota, matrix, dataset = load_dataset(
    'problems/instances/A1/symmetric/10.1.in')

A = np.matrix(matrix, dtype=float)
G = nx.from_numpy_array(A)

start_time = time.perf_counter()

for i in G.nodes:
    G.nodes[i]['bonus'] = dataset[i].bonus
    G.nodes[i]['penalty'] = dataset[i].penalty
    G.nodes[i]['id'] = dataset[i].id

finish_time = time.perf_counter()

duration = finish_time - start_time

print('duration - ', duration)

nx.draw(G, with_labels=True)

# MEMETICO

with open('memetic_history.txt', 'w') as file:
    file.write('iteration;duration(s);cost;route\n')

    for i in range(1):
        start_time = time.perf_counter()
        route = gen.memetic_algorithm(G, quota)
        finish_time = time.perf_counter()
        cost = route_cost(route, G)

        duration = finish_time - start_time

        route_ids = list(map(lambda node: node['id'], route))

        file.write(f'{i+1};{duration:.2f};{cost};{route_ids}\n')

# start_time = time.perf_counter()
# route = gen.memetic_algorithm(G, quota)
# finish_time = time.perf_counter()
# cost = route_cost(route, G)

# duration = finish_time - start_time
# report(duration, route, G)
# plot(route, G)
