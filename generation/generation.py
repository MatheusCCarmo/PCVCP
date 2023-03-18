from aux_functions import *
from local_search import local_search as ls
from entities.chromossome import Chromossome as Chromo
import random
import math
import time


# 06/01
# Estado da arte
# Trazer instancias e algoritmos da literatura
# Duvidas sobre IRACE, teste estatístico e algoritmos de comparação


# teste estatistico 
  # teste T
  # teste de mann-whitney U


#1 Introdução
# 2 Definição formal do problema
# 3 O estado da arte
# 4 O algoritmo desenvolvido
# 5 Experimentos
#   5.1 Metodologia 
#         5.1.1 Descrever as características das instâncias
#     5.2 Resultados para instâncias simétricas
#    5.3 Resultados para instâncias assimétricas
# 6 Conclusão
# 7 Referências Bibliográficas
# 10 instâncias simétricas e 10 assimétricas

# https://www.sciencedirect.com/
# https://link.springer.com/
# https://sci-hub.se/
# https://ieeexplore.ieee.org/Xplore/home.jsp
# https://dl.acm.org/

# Ver Algoritmos para comparação utilizados pelo estado da arte, selecionar 2 mais recentes

# Realizar teste estatístico,
# Dois testes estatísticos importantes:
# Kruskal-Wallis (utilizando para comparar mais de dois algoritmos ao mesmo tempo)
# Mann-Whitney (utilizando para comparar  APENAS dois algoritmos ao mesmo tempo)
# Nível de significância: 0.05
# p-valor<0.05, então significa que a primeira amostra (A) é melhor que a segunda (B).
# se o p-valor>=0.95, então a amostra B é conclusivamente melhor
# se o p-valor >=0.05 e p-valor<=0.95, então o resutlado é inconclusivo
# Reportar em tabelas no relatório (pode ser em apêndice)

# TODO: fazer experimentos para descobrir quais valores para os parametros apresenta os melhores resultados, a serem utilizados no experimento final
# ou
# usar o IRACE, IRACE recebe o intervalo dos parâmetros, o executável do algoritmo e ALGUMAS instâncias representativas. Devolve os melhores valores para os parâmetros.

#  Não utilizar no IRACE e nos experimentos as mesmas instancias para o trabalho 


# Parâmetros do genético:
# Taxa de mutação, taxa de recombinação (ou taxa de cruzamento), tamanho da população


population_size = 100
mutation_rate = 0.2
recombination_rate = 0.2

million = 200000

# enquanto critério de parada não for satisfeito faça:
    # para cada i de 1 até N (onde N é o tamanho da população) faça:
        # sorteie os pais. Aplique operadores (recombination, mutation).
        # busca local do resultante da mutação
    # busca local de alguns filhos gerados
    # combinar populações de pais e filhos
# retornar o melhor indivíduo

# usar GUROBI


# ALGORITTMOS


def memetic_algorithm(G, quota):
    population = init_population(G, quota) 

    # enquanto critério de parada não for satisfeito faça:
    while(route_cost.counter < million):

        children=[]
        # para cada i de 1 até N (onde N é o tamanho da população) faça:
        for i in range(population_size):

            # sorteie os pais. Aplique operadores (recombination, mutation).
            parent_1 = parents_selection(population, 4)
            parent_2 = parents_selection(population, 4)


            # aplicar recombinação
            rand = random.random()
            if(rand < recombination_rate):
                child_1, child_2 = recombination(parent_1, parent_2, G)  
                children.append(child_1)
                children.append(child_2)

            # aplicar mutação
            rand = random.random()
            if(rand < mutation_rate):
                mutated = mutation(population, G)
                # busca local do resultante da mutação
                new_mutated = local_search(mutated, G)

                children.append(new_mutated)

        # busca local de alguns filhos gerados
        random_children = random.choices(population, k=3)
        new_child_1 = local_search(random_children[0], G)

        new_child_2 = local_search(random_children[1], G)

        new_child_3 = local_search(random_children[2], G)


        children.append(new_child_1)
        children.append(new_child_2)
        children.append(new_child_3)
        
        new_population = [*population, *children]

        new_population.sort(key=lambda item: item.fitness_value(), reverse=True)

        population = new_population[:population_size]


    #buscar melhor solução da população
    best = best_genetic_solution(population)
    route_cost.counter = 0

    return best

def genetic_algorithm(G, quota):
    population = init_population(G, quota) 

    # enquanto critério de parada não for satisfeito faça:
    while(route_cost.counter < million):

        children=[]
        # para cada i de 1 até N (onde N é o tamanho da população) faça:
        for i in range(population_size):

            # sorteie os pais. Aplique operadores (recombination, mutation).
            parent_1 = parents_selection(population, 4)
            parent_2 = parents_selection(population, 4)


            # aplicar recombinação
            rand = random.random()
            if(rand < recombination_rate):
                child_1, child_2 = recombination(parent_1, parent_2, G)  
                children.append(child_1)
                children.append(child_2)

            # aplicar mutação
            rand = random.random()
            if(rand < mutation_rate):
                mutated = mutation(population, G)
                # busca local do resultante da mutação
                new_mutated = local_search(mutated, G)

                children.append(mutated)
        
        new_population = [*population, *children]
        new_population.sort(key=lambda item: item.fitness_value(), reverse=True)
        population = new_population[:population_size]


    #buscar melhor solução da população
    best = best_genetic_solution(population)
    route_cost.counter = 0

    return best


def grasp_algorithm(quota, G):
    grasp_solutions = []
    alfa_grasp = 0.1
    best_cost = math.inf

    while(route_cost.counter < million):

        route = grasp_construction(G, quota, alfa_grasp)
        
        if(route in grasp_solutions):
            continue
        grasp_solutions.append(route)

        route = ls.drop_step(route, quota, G)

        route = ls.swap_2_opt(route, G)
        
        cost = route_cost(route, G)

        if(cost < best_cost):
            best_route = route
            best_cost = cost


    route_cost.counter = 0
    return best_route



# GENETIC AUX

def init_population(G, quota):
    population = []
    for i in range(population_size):
        route = generate_random_route(G, quota)
        population.append(Chromo(route, G))
    return population

def evaluate(population):

    fitness_sum = 0
    for i in range(len(population)):
        fitness_sum += population[i].fitness_value

    return fitness_sum/len(population)

def parents_selection(population, k):
    candidates = random.choices(population, k=k)
    candidates.sort(key=lambda item: item.fitness_value(), reverse=True)

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
    #TODO: adicionartaxa de mutação
    choosen = random.choice(population)

    first, second = random.sample(range(1,len(choosen.route)), 2)
 
    i = min(first, second)
    j = max(first, second)

    new_route = swap_2(i, j, choosen.route)
    new_chormo = Chromo(new_route, G)
    return new_chormo

def local_search(chromo, G):
    new_route = ls.swap_2_opt(chromo.route, G)
    new_chromo = Chromo(new_route, G)

    return new_chromo

def best_genetic_solution(population):
    best_chromo = population[0]
    for i in range(population_size):
        fitness_value = population[i].fitness_value()
        if(fitness_value > best_chromo.fitness_value()):
            best_chromo = population[i]
    return best_chromo.route


def generate_random_route(G, quota):
    bonus_colected = 0
    route = [G.nodes[0]]
    while(bonus_colected < quota):
        random_i = random.randrange(0, len(G.nodes))
        if(G.nodes[random_i] not in route):
            bonus_colected += G.nodes[random_i]['bonus']
            route.append(G.nodes[random_i])
            
    return route


# CONSTRUCT AUX

def drop_step(route, quota, G):
    bonus_colected = calculate_bonus_colected(route, G)
    best_economy = -math.inf

    improved = True
    
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
            k_economy_value = k_edge1['weight'] + k_edge2['weight'] - edge['weight'] - G.nodes[j]['penalty']
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



# def add_step(G, quota, alfa_grasp):    
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
#                     k_economy_value = edge['weight'] + G.nodes[k]['penalty'] - k_edge1['weight'] - k_edge2['weight']
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
                k_economy_value = edge['weight'] + G.nodes[k]['penalty'] - k_edge1['weight'] - k_edge2['weight']
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



# OTHER ALGORITHMS

# Algoritmo GRASP + VNS.
def grasp_vns(G, quota):
    max_iter = 4000
    alfa_grasp = 0.2

    # Construção
    route = [G.nodes[0]]
    best_route = route
    bonus_colected = calculate_bonus_colected(route, G)

    for n in range(max_iter):
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
                k_economy_value = edge['weight'] + G.nodes[k]['penalty'] - k_edge1['weight'] - k_edge2['weight']
                economy_list.append((k,k_economy_value))
        has_positive_economy = any(economy > 0 for economy in economy_list)

        while(bonus_colected < quota or has_positive_economy):
            economy_list.sort(key=lambda item: item[1],reverse=True)
            best_economy = economy_list[0][1]
            worst_economy = economy_list[-1][1]
            grasp_tsh = best_economy - alfa_grasp * (best_economy - worst_economy)
            grasp_candidates = list(filter(lambda item: item[1] >= grasp_tsh, economy_list))
            insertion_selected = random.choice(grasp_candidates)
            if(best_economy > 0 or bonus_colected < quota):
                route.append(G.nodes[insertion_selected[0]])
                economy_list.remove(insertion_selected)
            # TODO: Analisar se precisa recaul
            bonus_colected = calculate_bonus_colected(route, G)
            has_positive_economy = any(economy > 0 for economy in economy_list)

        cost = route_cost(route, G)
        if(cost < best_cost):
            best_cost = cost
            best_route = route
    route = best_route

    # VNS
    no_improvement_time = 0
    neighboors = [neighboor_1(route, G), 
                  neighboor_2(route, G), 
                  neighboor_3(route), 
                  neighboor_4(route), 
                  neighboor_5(route, G)]
    neighboors_len = len(neighboors)
    while(no_improvement_time < max_time):
        init_time = time.perf_counter()
        choosen = random.choice(neighboors)

        # VND
        vnd_route = vnd(choosen)
        
        vnd_cost = route_cost(route, G)
        
        if(vnd_cost < best_cost):
            best_cost = vnd_cost
            best_route = vnd_route

        current_time = time.perf_counter()
        no_improvement_time = current_time - init_time


    # Busca Local


def vns(route, G):    
    max_time = 200
    
    no_improvement_time = 0
    neighboors = [neighboor_1(route, G), 
                  neighboor_2(route, G), 
                  neighboor_3(route, G), 
                  neighboor_4(route, G), 
                  neighboor_5(route, G)]
    neighboors_len = len(neighboors)

    while(no_improvement_time < max_time):
        init_time = time.perf_counter()
        k = 0

        while(k < neighboors_len):
            choosen_function = neighboors[k]

            neighboor = choosen_function(route, G)

            # VND
            vnd_route = vnd(neighboor)
            
            vnd_cost = route_cost(route, G)
            
            if(vnd_cost < best_cost):
                best_cost = vnd_cost
                best_route = vnd_route
            else:
                k += 1

            current_time = time.perf_counter()
            no_improvement_time = current_time - init_time

        return best_route

# TODO:
def vnd(route): 
    return route

# Retirar vértice de maior economia:
def neighboor_1(route, G):
    economy_list = []
    for r in range(len(route) - 1):
        i = route[r-1]['id']
        j = route[r]['id']
        s = route[r+1]['id']
        k_edge1 = G.edges[i,j]
        k_edge2 = G.edges[j,s]
        edge = G.edges[i,s]
        k_economy_value = k_edge1['weight'] + k_edge2['weight'] - edge['weight'] - G.nodes[j]['penalty']
        economy_list.append((j,k_economy_value))

    economy_list.sort(key=lambda item: item[1],reverse=True)
    best_economy_item = economy_list[0]
    route.remove(G.nodes[best_economy_item[0]])        
    return route

# Inserir vértice de maior economia:
def neighboor_2(route, G):
    economy_list = []
    for r in range(len(route) - 1):
        i = route[r-1]['id']
        j = route[r]['id']
        s = route[r+1]['id']
        k_edge1 = G.edges[i,j]
        k_edge2 = G.edges[j,s]
        edge = G.edges[i,s]
        k_economy_value = k_edge1['weight'] + k_edge2['weight'] - edge['weight'] - G.nodes[j]['penalty']
        economy_list.append((j,k_economy_value,r))

    economy_list.sort(key=lambda item: item[1],reverse=True)
    best_economy_item = economy_list[0]
    route.insert(best_economy_item[2],G.nodes[best_economy_item[0]])        
    
    return route

# Troca dois vertices:
def neighboor_3(route, G):
    first, second = random.sample(range(1,len(route)), 2)
 
    i = min(first, second)
    j = max(first, second)
    route = swap_2(i, j, route)     
      
    return route

# Remove vertice aleatorio:
def neighboor_4(route, G):
    random_i = random.randrange(0, len(route))

    del route[random_i]
      
    return route

# Insere vertice aleatorio:
def neighboor_5(route, G):
    random_node_i = random.randrange(0, len(G.nodes))
    while(random_node_i in route):
        random_node_i = random.randrange(0, len(G.nodes))
    random_i = random.randrange(0, len(route))

    route.insert(random_i, random_node_i)
      
    return route