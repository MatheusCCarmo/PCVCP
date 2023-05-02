import networkx as nx
import numpy as np
import time
import cProfile

from aux_functions import *
from generation import generation as gen
from local_search import local_search as ls

n, quota, matrix, dataset = load_dataset(
    'problems/instances/A1/symmetric/10.1.in')

A = np.matrix(matrix, dtype=float)

for i in range(len(A)):
    for j in range(len(A)):
        if A[i, j] == 0:
            A[i, j] = dataset[i].penalty

G = nx.from_numpy_array(A)

for i in G.nodes:
    G.nodes[i]['bonus'] = dataset[i].bonus
    G.nodes[i]['penalty'] = dataset[i].penalty


route = [0, 6, 4, 7, 5]


route_edges = [(route[i], route[i+1])
               for i in range(len(route) - 1)]
route_edges.append((route[-1], route[0]))

# Get the number of nodes in the route
num_nodes = len(G.nodes)

# Initialize the matrix with zeros
edges_incidence = np.zeros((num_nodes, num_nodes))


node_incidence = [1 if i in route else 0 for i in range(len(G.nodes))]

for edge in route_edges:
    edges_incidence[edge[0], edge[1]] = 1

# # xii = 1 - yi
for i in G.nodes:
    edges_incidence[i, i] = 1 - node_incidence[i]


# def route_cost(route, G):
#     route_cost.counter += 1
#     penalties = calculate_penalties(route, G)
#     distance = calculate_route_distance(route, G)
#     cost = penalties + distance
#     return cost


def route_cost(route, G):
    route_cost.counter += 1
    cost = 0

    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            cost += G[i][j]['weight'] * edges_incidence[i, j]

    return cost


route_cost.counter = 0


def test_algorithm(G, quota):
    for i in range(10000):
        cost = route_cost(route, G)
        # cost = 0
        # for i in range(10):
        #     for j in range(10):
        #         cost += 2
        continue
    print(cost)
    return

# MEMETICO

def main():
    for i in range(1):
        start_time = time.perf_counter()
        test_algorithm(G, quota)
        finish_time = time.perf_counter()

        duration = finish_time - start_time

        print('duration - ', duration)


if __name__ == '__main__':
    print('Timing program...')
    cProfile.run('main()', sort='tottime')
