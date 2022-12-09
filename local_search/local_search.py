import networkx as nx
import matplotlib.pyplot as plt
from aux_functions import *


def swap(i, j, route):
    route_A = route[:i]
    route_B = route[i:j]
    route_B.reverse()
    new_route = [*route_A,*route_B, *route[j:]]
    return new_route

def swap_2_opt(route, G, my_pos):
    cur_length = calculate_route_distance(route, G)

    improved = True
    bonus_label = bonus_labels(route)

    while improved:
        improved = False
        routeLen = len(route)
        for i in range(routeLen):
            for j in range(i+1,routeLen):
                
                
                new_route = swap(i,j, route)
                new_length = calculate_route_distance(new_route, G)
                
                # update the route, if improved
                if new_length < cur_length:
                    cur_length = new_length
                    route = new_route
                    improved = True
                    bonus_label = bonus_labels(route)
                    
                    # draw the new tour
                    route_edges = [ (route[i-1]['id'],route[i]['id']) for i in range(len(route)) ]
                    plt.figure() # call this to create a new figure, instead of drawing over the previous one(s)
                    nx.draw(G.edge_subgraph(route_edges), pos=my_pos, with_labels=True)
                    # plt.figure()
                    # nx.draw(G.edge_subgraph(route_edges), pos=my_pos)
                    # nx.draw_networkx_labels(G.edge_subgraph(route_edges),  pos=my_pos, labels=bonus_label, font_size=10, font_color="whitesmoke")
    return route