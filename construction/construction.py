import networkx as nx
import matplotlib.pyplot as plt
from aux_functions import *

infinity = 9999999999

def insert_from_closest(G, quota, my_pos):
    # find closest vertex from node 0
    closest_vertex = 0
    closest_distance = infinity
    for i,j in G.edges:
        if(i>0):
            break
        if G.edges[i,j]['length'] < closest_distance:
            closest_vertex = j
            closest_distance = G.edges[i,j]['length']
        
    # init route
    route = [G.nodes[0], G.nodes[closest_vertex]]

    bonus_colected = calculate_bonus_colected(route, G)

    # insert
    while bonus_colected < quota:
        k_best_economy_value = -infinity
        k_best_economy = 0
        i_best_economy = 0
        for k in range(len(G.nodes)):
            if G.nodes[k] not in route:
                for r in range(len(route)):
                    if r < len(route) -1:
                        i = route[r]['id']
                        j = route[r + 1]['id']
                    else:
                        i = route[r]['id']
                        j = route[0]['id']
                    edge = G.edges[i,j]
                    k_edge1 = G.edges[i,k]
                    k_edge2 = G.edges[k,j]
                    k_economy_value = edge['length'] + G.nodes[k]['penalty'] - k_edge1['length'] - k_edge2['length']
                    if k_economy_value > k_best_economy_value:
                        k_best_economy_value = k_economy_value
                        i_best_economy = i
                        k_best_economy = k
        bonus_colected = calculate_bonus_colected(route, G)
        if bonus_colected < quota and k_best_economy != 0:
            route.insert(i_best_economy + 1, G.nodes[k_best_economy])
        else:
            if k_best_economy_value <= 0:
                break
        plt.figure()
        route_edges = [ (route[i-1]['id'],route[i]['id']) for i in range(len(route)) ]
        nx.draw(G.edge_subgraph(route_edges), pos=my_pos, with_labels=True)

    return route