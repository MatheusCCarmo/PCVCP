import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from entities.vertex import Vertex

calculations_counter = 0


def eucl_dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def swap_2(i, j, route):
    route_A = route[:i]
    route_B = route[i:j]
    route_B.reverse()
    new_route = [*route_A, *route_B, *route[j:]]
    # new_route = route[:i+1] + list(reversed(route[i+1:j+1])) + route[j+1:]
    return new_route


def calculate_route_distance(route, G):

    distance = 0
    # print(route)
    for v in range(len(route)):
        i = route[v - 1]['id']
        j = route[v]['id']
        distance += G.edges[i, j]['weight']
    return distance


def calculate_penalties(route, G):

    penalties = 0
    for i in range(len(G.nodes)):
        if G.nodes[i] not in route:
            penalties += G.nodes[i]['penalty']

    return penalties


def calculate_bonus_colected(route, G):

    bonus = 0
    for r in route:
        bonus += G.nodes[r['id']]['bonus']

    return bonus


def bonus_labels(route):
    bonus_label = {}
    for i in range(len(route)):
        bonus_label[route[i]['id']] = route[i]['bonus']
    return bonus_label


def load_dataset(file_name):
    n = 0
    quota = 0
    matrix = []
    dataset = []
    with open(file_name, "r") as f:
        for line in f:
            # remove spaces at the beginning and the end if they are available
            new_line = line.strip()

            new_line = new_line.split(" ")  # split a string into a list
            new_line = list(filter(lambda x: x != '', new_line))

            if (len(new_line) == 0):
                continue

            if (n == 0):
                # check dataset file to see why id,x,y = 0,1,2
                n = int(new_line[0])
                continue

            if (len(matrix) < n):
                float_line = list(
                    map(lambda wheight: float(wheight), new_line))
                matrix.append(float_line)
                continue

            if (len(new_line) == 3):
                # check dataset file to see why id,x,y = 0,1,2
                id, bonus = new_line[0], int(new_line[1])
                dataset.append(Vertex(id=id, bonus=bonus))
                continue

            if (len(new_line) == 1):
                quota = int(new_line[0])
                continue

    return n, quota, matrix, dataset


def load_other_dataset(file_name):
    n = 0
    quota = 0
    matrix = []
    node_prizes = []
    node_penalties = []
    is_coord = False
    is_prize = False
    is_penalty = False

    with open(file_name, "r") as f:
        for line in f:
            # remove spaces at the beginning and the end if they are available
            new_line = line.strip()

            new_line = new_line.split(" ")  # split a string into a list
            new_line = list(filter(lambda x: x != '', new_line))

            if (len(new_line) == 0):
                continue

            if (is_prize):
                node_prizes = list(map(lambda prize: int(prize), new_line))
                is_prize = False

            if (is_penalty):
                node_penalties = list(
                    map(lambda penalty: int(penalty), new_line))
                is_penalty = False

            if (new_line[0] == "PRIZE"):
                is_prize = True
                continue

            if (new_line[0] == "TRAVEL"):
                is_coord = True
                continue

            if (new_line[0] == "PENALTY"):
                is_penalty = True
                continue

            n = len(node_prizes)

            if (is_coord):
                if (len(matrix) < n):
                    float_line = list(
                        map(lambda wheight: float(wheight), new_line))
                    matrix.append(float_line)
                    continue
                else:
                    is_coord = False

            quota = sum(node_prizes) * 0.75

    return node_prizes, node_penalties, matrix, quota

# def load_symmetric(file_name):
#     dataset = []
#     is_coord = False
#     with open(file_name, "r") as f:
#         for line in f:
#             new_line = line.strip()  # remove spaces at the beginning and the end if they are available
#             if(new_line == 'NODE_COORD_SECTION' or new_line == 'DISPLAY_DATA_SECTION'):
#                 is_coord = True
#                 continue
#             elif(new_line == 'EOF'):
#                 break
#             if(is_coord):
#                 new_line = new_line.split(" ")  # split a string into a list
#                 new_line = list(filter(lambda x: x != '', new_line))
#                 id, x, y = new_line[0], new_line[1], new_line[2]  # check dataset file to see why id,x,y = 0,1,2
#                 dataset.append(Vertex(id=id, x=x, y=y))  # Create a Node object with id, x, y and add to the data list
#     return dataset

# def load_asymmetric(file_name):
#     dataset = []
#     is_coord = False
#     with open(file_name, "r") as f:
#         for line in f:
#             new_line = line.strip()  # remove spaces at the beginning and the end if they are available
#             if(new_line == 'NODE_COORD_SECTION' or new_line == 'DISPLAY_DATA_SECTION'):
#                 is_coord = True
#                 continue
#             elif(new_line == 'EOF'):
#                 break
#             if(is_coord):
#                 new_line = new_line.split(" ")  # split a string into a list
#                 new_line = list(filter(lambda x: x != '', new_line))
#                 id, x, y = new_line[0], new_line[1], new_line[2]  # check dataset file to see why id,x,y = 0,1,2
#                 dataset.append(Vertex(id=id, x=x, y=y))  # Create a Node object with id, x, y and add to the data list
#     return dataset


# def route_cost(route, G):
#     route_cost.counter += 1
#     penalties = calculate_penalties(route, G)
#     distance = calculate_route_distance(route, G)
#     cost = penalties + distance

#     return cost

def route_cost(route, G):
    route_cost.counter += 1

    route_edges = [(route[i-1], route[i])
                   for i in range(len(route))]

    num_nodes = len(G.nodes)

    # Initialize the matrix with zeros
    edges_incidence = np.zeros((num_nodes, num_nodes))

    node_incidence = [
        1 if i in route else 0 for i in range(len(G.nodes))]

    # Set the entries in the matrix to 1 for the route edges
    for edge in route_edges:
        edges_incidence[edge[0], edge[1]] = 1

    # xii = 1 - yi
    for i in G.nodes:
        edges_incidence[i, i] = 1 - node_incidence[i]

    cost = 0

    # obter as posições onde há valor 1
    indices = np.where(edges_incidence == 1)
    for i, j in zip(indices[0], indices[1]):
        cost += G[i][j]['weight']

    return cost


route_cost.counter = 0


def report(duration, route, G):
    print(f"Finished in {duration:0.4f} seconds")
    cost = route_cost(route, G)
    bonus = calculate_bonus_colected(route, G)
    print('Cost:', cost)
    print('Bonus Colected:', bonus)


def plot(route, G):
    plt.figure()
    route_edges = [(route[i-1]['id'], route[i]['id'])
                   for i in range(len(route))]
    nx.draw(G.edge_subgraph(route_edges), with_labels=True)
