from aux_functions import *
import random

infinity = 9999999999

generations_size = 200
population_size = 100
mut_rate = 0.2


def insert_from_closest(G, quota):
    route = [G.nodes[0]]

    bonus_colected = calculate_bonus_colected(route, G)
    k_best_economy_value = -infinity



    # insert
    while bonus_colected < quota or k_best_economy_value > 0:
        k_best_economy_value = -infinity
        k_best_economy = 0
        for k in range(len(G.nodes)):
            if G.nodes[k] not in route:
                if(len(route) == 1):
                    route.insert(1, G.nodes[k])
                    continue
                for r in range(len(route)):
                    i = route[r-1]['id']
                    j = route[r]['id']
                    edge = G.edges[i,j]
                    k_edge1 = G.edges[i,k]
                    k_edge2 = G.edges[k,j]
                    k_economy_value = edge['length'] + G.nodes[k]['penalty'] - k_edge1['length'] - k_edge2['length']

                    if k_economy_value > k_best_economy_value:
                        k_best_economy_value = k_economy_value
                        k_best_economy = k
                        r_best_economy = r        
        
        if(k_best_economy_value > 0 or bonus_colected < quota):
            route.insert(r_best_economy, G.nodes[k_best_economy])
        bonus_colected = calculate_bonus_colected(route, G)
    return route


def genetic_algorithm(G, quota):
    # gerar uma população de soluçõs
    population = []
    for i in range(population_size):
        population.append(init_population(G, quota))

    # avaliar soluções geradas
    evaluate(population)

    for i in range(generations_size):
        # selecionar um conjunto de pais
        father_selection(population)

        # realizar cruzamento de k pais com uma dada probabilidade
        crossover(population)

        # realizar mutação das soluções geradas
        new_population = mutation(population)

        # avaliar a aptidao das soluções geradas
        evaluate(population)

        # atualizar a população
        population = new_population

    #buscar melhor solução da população
    best = best_solution(population)
    return best



def init_population(population):
    return

def evaluate(population):
    return

def father_selection(population):
    return

def crossover(population):
    return

def mutation(population):
    return

def best_solution(population):
    return


def generate_random_route(G, quota):
    bonus_colected = 0
    route = [G.nodes[0]]
    while(bonus_colected < quota):
        random_i = random.randint(0, len(G.nodes) - 1)
        if(G.nodes[random_i] not in route):
            bonus_colected += G.nodes[random_i]['bonus']
            route.append(G.nodes[random_i])
    return route




def grasp_construction(G, quota, alfa_grasp):
    
    route = [G.nodes[0]]

    bonus_colected = calculate_bonus_colected(route, G)
    k_best_economy_value = -infinity

    # insert
    while bonus_colected < quota or k_best_economy_value > 0:
        k_best_economy_value = -infinity
        economy_list = []
        for k in range(len(G.nodes)):
            if G.nodes[k] not in route:
                if(len(route) == 1):
                    route.insert(1, G.nodes[k])
                    continue
                for r in range(len(route)):
                    i = route[r-1]['id']
                    j = route[r]['id']
                    edge = G.edges[i,j]
                    k_edge1 = G.edges[i,k]
                    k_edge2 = G.edges[k,j]
                    k_economy_value = edge['length'] + G.nodes[k]['penalty'] - k_edge1['length'] - k_edge2['length']
                    economy_list.append((k,k_economy_value,r))
                    if k_economy_value > k_best_economy_value:
                        k_best_economy_value = k_economy_value
        economy_list.sort(key=lambda item: item[1],reverse=True)
        best_economy = economy_list[0][1]
        worst_economy = economy_list[-1][1]
        grasp_tsh = best_economy - alfa_grasp * (best_economy - worst_economy)
        grasp_candidates = list(filter(lambda item: item[1] >= grasp_tsh, economy_list))
        insertion_selected = random.choice(grasp_candidates)
        if(best_economy > 0 or bonus_colected < quota):
            route.insert(insertion_selected[2], G.nodes[economy_list[0][0]])
        bonus_colected = calculate_bonus_colected(route, G)
    return route