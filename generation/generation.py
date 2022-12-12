from aux_functions import *
from entities.chromossome import Chromossome as Chromo
import random
import math
import time

generations_size = 100000
population_size = 200


def insert_from_closest(G, quota):
    route = [G.nodes[0]]

    bonus_colected = calculate_bonus_colected(route, G)
    k_best_economy_value = -math.inf



    # insert
    while bonus_colected < quota or k_best_economy_value > 0:
        k_best_economy_value = -math.inf
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
    population = init_population(G, quota) 

    # avaliar soluções geradas
    # evaluate(population)
    
    populations_fitness = []
    start_time = time.time()
    # for i in range(generations_size):

    while(time.time() - start_time < 30):
        # selecionar um conjunto de pais
        parent_1 = parents_selection(population, 4)
        parent_2 = parents_selection(population, 4)
 
        # realizar cruzamento de k pais com uma dada probabilidade
        child_1, child_2 = recombination(parent_1, parent_2, G)

        # realizar mutação das soluções geradas
        population = mutation(population, G)

        # avaliar a aptidao das soluções geradas
        populations_fitness.append(evaluate(population))

        # atualizar a população
        population.sort(key=lambda item: item.fitness_value, reverse=True)
        insert_index = random.randint(population_size/2, population_size - 1)
        population[insert_index] = child_1
        insert_index = random.randint(population_size/2, population_size - 1)
        population[insert_index] = child_2
    
    #buscar melhor solução da população
    best = best_solution(population)
    return best



def init_population(G, quota):
    population = []
    for i in range(population_size):
        route = grasp_construction(G, quota, 0.1)
        route = drop_step(route, quota, G)
        population.append(Chromo(route, G))
    return population

def evaluate(population):
    fitness_sum = 0
    for i in range(len(population)):
        fitness_sum += population[i].fitness_value
    return fitness_sum/len(population)

def parents_selection(population, k):
    #tournament
    candidates = random.choices(population, k=k)
    candidates.sort(key=lambda item: item.fitness_value, reverse=True)

    return candidates[0]
 
def recombination(parent_1, parent_2, G):
    child_1, child_2 = crossover_two(parent_1, parent_2)

    chromo_1 = Chromo(child_1, G)
    chromo_2 = Chromo(child_2, G)

    return chromo_1, chromo_2

def crossover_two(parent_1, parent_2):  # two points crossover
    len_min =  begin = min(len(parent_1.route), len(parent_2.route))
    point_1, point_2 = random.sample(range(1, len_min-1), 2)
    begin = min(point_1, point_2)
    end = max(point_1, point_2)

    child_1 = parent_1.route[begin:end]
    child_2 = parent_2.route[begin:end]


    child_1_remain = [item for item in parent_2.route[1:] if item not in child_1]
    child_2_remain = [item for item in parent_1.route[1:] if item not in child_2]


    child_1 += child_1_remain
    child_2 += child_2_remain

    child_1.insert(0, parent_1.route[0])

    child_2.insert(0, parent_2.route[0])

    return child_1, child_2

def mutation(population, G):
    choosen = random.choice(population)
    population.remove(choosen)

    first, second = random.sample(range(1,len(choosen.route)), 2)
 
    i = min(first, second)
    j = max(first, second)

    new_route = swap_2(i, j, choosen.route)
    new_chormo = Chromo(new_route, G)
    population.append(new_chormo)
    return population

def best_solution(population):
    best_chromo = population[0]
    for i in range(population_size):
        fitness_value = population[i].fitness_value
        if(fitness_value > best_chromo.fitness_value):
            best_chromo = population[i]
    return best_chromo.route


def generate_random_route(G, quota):
    bonus_colected = 0
    route = [G.nodes[0]]
    while(bonus_colected < quota):
        random_i = random.randint(0, len(G.nodes) - 1)
        if(G.nodes[random_i] not in route):
            bonus_colected += G.nodes[random_i]['bonus']
            route.append(G.nodes[random_i])
            
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
            k_edge1 = G.edges[i,j]
            k_edge2 = G.edges[j,s]
            edge = G.edges[i,s]
            k_economy_value = k_edge1['length'] + k_edge2['length'] - edge['length'] - G.nodes[j]['penalty']
            economy_list.append((j,k_economy_value,r))
        if(len(economy_list) == 0):
            continue 
        economy_list.sort(key=lambda item: item[1],reverse=True)
        best_economy_item = economy_list[0]
        best_economy = best_economy_item[1]
        item_bonus = G.nodes[best_economy_item[0]]['bonus']
        if(best_economy > 0 and (bonus_colected - item_bonus) > quota):
            route.remove(G.nodes[best_economy_item[0]])
            improved = True
        else:
            improved = False
        bonus_colected = calculate_bonus_colected(route, G)
        
    return route


# def grasp_construction(G, quota, alfa_grasp):
    
#     route = [G.nodes[0]]

#     bonus_colected = calculate_bonus_colected(route, G)
#     best_economy = -math.inf

#     # insert
#     while bonus_colected < quota or best_economy > 0:
#         best_economy = -math.inf
#         economy_list = []
#         for k in range(len(G.nodes)):
#             if G.nodes[k] not in route:
#                 if(len(route) == 1):
#                     route.insert(1, G.nodes[k])
#                     continue
#                 for r in range(len(route)):
#                     i = route[r-1]['id']
#                     j = route[r]['id']
#                     edge = G.edges[i,j]
#                     k_edge1 = G.edges[i,k]
#                     k_edge2 = G.edges[k,j]
#                     k_economy_value = edge['length'] + G.nodes[k]['penalty'] - k_edge1['length'] - k_edge2['length']
#                     economy_list.append((k,k_economy_value,r))
#         if(len(economy_list) == 0):
#             continue 
#         economy_list.sort(key=lambda item: item[1],reverse=True)
#         best_economy = economy_list[0][1]
#         worst_economy = economy_list[-1][1]
#         grasp_tsh = best_economy - alfa_grasp * (best_economy - worst_economy)
#         grasp_candidates = list(filter(lambda item: item[1] >= grasp_tsh, economy_list))
#         insertion_selected = random.choice(grasp_candidates)
#         if(best_economy > 0 or bonus_colected < quota):
#             route.insert(insertion_selected[2], G.nodes[insertion_selected[0]])
#         bonus_colected = calculate_bonus_colected(route, G)
#     return route



def grasp_construction(G, quota, alfa_grasp):
    
    route = [G.nodes[0]]

    bonus_colected = calculate_bonus_colected(route, G)
    best_economy = -math.inf

    # insert
    while bonus_colected < quota or best_economy > 0:
        best_economy = -math.inf
        economy_list = []
        for k in range(len(G.nodes)):
            if G.nodes[k] not in route:
                route_len = len(route)
                if(route_len == 1):
                    route.append(G.nodes[k])
                    continue
                i = route[-2]['id']
                j = route[-1]['id']
                edge = G.edges[i,j]
                k_edge1 = G.edges[i,k]
                k_edge2 = G.edges[k,j]
                k_economy_value = edge['length'] + G.nodes[k]['penalty'] - k_edge1['length'] - k_edge2['length']
                economy_list.append((k,k_economy_value))
                   
        if(len(economy_list) == 0):
            continue 
        economy_list.sort(key=lambda item: item[1],reverse=True)
        best_economy = economy_list[0][1]
        worst_economy = economy_list[-1][1]
        grasp_tsh = best_economy - alfa_grasp * (best_economy - worst_economy)
        grasp_candidates = list(filter(lambda item: item[1] >= grasp_tsh, economy_list))
        insertion_selected = random.choice(grasp_candidates)
        if(best_economy > 0 or bonus_colected < quota):
            route.append(G.nodes[insertion_selected[0]])
        bonus_colected = calculate_bonus_colected(route, G)
    return route


# memetico
#     Entrada: una instancia I de un problema P.
# Salida: una solución sol.
# // generar población inicial
# 1 : para j ← 1:popsize hacer
# 2 : sea ind ← GenerarSolucionHeuristica (I)
# 3 : sea pop[j] ← MejoraLocal (ind, I)
# 4 : finpara
# 5 : repetir // bucle generacional
# // Selección
# 6 : sea criadores ← SeleccionarDePoblacion (pop)
# // Reproducción segmentada
# 7 : sea auxpop[0] ← pop
# 8 : para j ← 1:#op hacer
# 9 : sea auxpop[j] ← AplicarOperador (op[j], auxpop[j ¡ 1], I)
# 10 : finnpara
# 11 : sea newpop ← auxpop[#op]
# // Reemplazo
# 12 : sea pop ← ActualizarProblacion (pop, newpop)
# // Comprobar convergencia
# 13 : si Convergencia (pop) entonces
# 14 : sea pop ← RefrescarPoblacion (pop, I)
# 15 : finsi
# 16 : hasta CriterioTerminacion (pop, I)
# 17 : devolver Mejor (pop, I)