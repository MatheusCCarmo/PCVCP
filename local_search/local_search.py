import networkx as nx
import matplotlib.pyplot as plt
from aux_functions import *
import math


def swap_2_opt(route, quota, G):
    cur_cost = route_cost(route, G)

    improved = True

    while improved:
        counter = 0
        improved = False
        routeLen = len(route)
        for i in range(1, routeLen):
            for j in range(i+1, routeLen):

                new_route = swap_2(i, j, route)
                new_cost = route_cost(new_route, G)

                # update the route, if improved
                if new_cost < cur_cost:
                    cur_cost = new_cost
                    route = new_route
                    improved = True
                    counter = 0

                counter += 1
                if counter > 100000:
                    break
    return route


def add_drop(route, quota, G):
    route = add_step(route, quota, G)
    route = drop_step(route, quota, G)
    return route


def seq_drop_seq_add(route, quota, G):
    route = drop_step(route, quota, G)
    route = add_step(route, quota, G)
    return route


def drop_step(route, quota, G):
    bonus_colected = calculate_bonus_colected(route, G)
    best_economy = -math.inf

    improved = True

    # insert
    while improved:
        best_economy = -math.inf
        economy_list = []
        for r in range(len(route) - 1):
            i = route[r-1]['id']
            j = route[r]['id']
            s = route[r+1]['id']
            k_edge1 = G.edges[i, j]
            k_edge2 = G.edges[j, s]
            edge = G.edges[i, s]
            k_economy_value = k_edge1['weight'] + k_edge2['weight'] - \
                edge['weight'] - G.nodes[j]['penalty']
            economy_list.append((j, k_economy_value, r))
        if (len(economy_list) == 0):
            continue
        economy_list.sort(key=lambda item: item[1], reverse=True)
        best_economy_item = economy_list[0]
        best_economy = best_economy_item[1]
        item_bonus = G.nodes[best_economy_item[0]]['bonus']
        if (best_economy > 0 and (bonus_colected - item_bonus) > quota):
            route.remove(G.nodes[best_economy_item[0]])
            improved = True
        else:
            improved = False
        bonus_colected = calculate_bonus_colected(route, G)

    return route


def add_step(route, quota, G):
    bonus_colected = calculate_bonus_colected(route, G)
    k_best_economy_value = -math.inf

    # insert
    while bonus_colected < quota or k_best_economy_value > 0:
        k_best_economy_value = -math.inf
        k_best_economy = 0
        for k in range(len(G.nodes)):
            if G.nodes[k] not in route:
                for r in range(len(route)):
                    i = route[r-1]['id']
                    j = route[r]['id']
                    edge = G.edges[i, j]
                    k_edge1 = G.edges[i, k]
                    k_edge2 = G.edges[k, j]
                    k_economy_value = edge['weight'] + G.nodes[k]['penalty'] - \
                        k_edge1['weight'] - k_edge2['weight']

                    if k_economy_value > k_best_economy_value:
                        k_best_economy_value = k_economy_value
                        k_best_economy = k
                        r_best_economy = r

        if (k_best_economy_value > 0 or bonus_colected < quota):
            route.insert(r_best_economy, G.nodes[k_best_economy])
        bonus_colected = calculate_bonus_colected(route, G)
    return route

# lin-kernighan
# def lin_kernighan(route, G):
#     route_lin = [*route, route[0]]

#     best_route = route_lin
#     for v1 in range(len(route) - 1):
#         i = route_lin[v1]['id']
#         j = route_lin[v1+1]['id']
#         edge1 = (i,j)
#         edge1_length = G.edges[i,j]['weight']
#         minimum_edge_length = edge1_length
#         minimum_edge = edge1
#         for v2 in range(v1, len(route_lin) - 1):
#             i = route_lin[v2]['id']
#             j = route_lin[v2 + 1]['id']
#             edge2 = (i,j)
#             edge2_length = G.edges[i,j]['weight']
#             if(edge2_length < minimum_edge_length):
#                 minimum_edge_length = edge2_length
#                 minimum_edge = edge2
#         if(minimum_edge != edge1):
#             delta = swap_edges(route_lin, edge1, edge2)
#             best_route = best_from(delta, best_route, G)
#     return best_route

# def swap_edges(route_lin, edge1, edge2):
#     route_lin = swap_delta(route_lin, )
#     return route_lin

# def swap_delta(delta, edge1, edge2):
#     [a,b,c,d,e,a]
#     [(a,b), (b,c), (c,d), (d,e), (e,a)]

#     [a,b,c,d,e,c]
#     [(a,b), (b,c), (c,d), (d,e), (e,c)]
#     new_route = [*delta[:len(delta) -1], delta[i]]
#     return route_lin


# def best_from(delta, route, G):
#     print(delta)
#     delta_repaired = delta[:len(route)]
#     print(delta_repaired)
#     delta_repaired_distance = calculate_route_distance(delta, G)
#     route_distance = calculate_lin_route_distance(route, G)
#     if delta_repaired_distance < route_distance:
#         route = delta_repaired
#     delta = best_delta_swap(delta)

#     delta_distance = calculate_lin_route_distance(delta, G)
#     if route_distance <= delta_distance:
#         return route
#     else:
#         return best_from(delta, route)

# def best_delta_swap(delta, G):
#     best_distance = calculate_lin_route_distance(delta, G)
#     for i in range(len(delta)):
#         new_route = [*delta[:len(delta) -1], delta[i]]
#         distance = calculate_lin_route_distance(delta)
#         if distance < best_distance:
#             best_distance = distance
#             delta = new_route
#     return delta

# def calculate_lin_route_distance(route_lin, G):
#     distance_cost = 0
#     for v in range(len(route_lin) - 1):
#         i = route_lin[v]['id']
#         j = route_lin[v + 1]['id']
#         distance_cost += G.edges[i,j]['weight']
#     return distance_cost
