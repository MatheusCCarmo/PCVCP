from aux_functions import *
from local_search import local_search as ls
from entities.chromossome import Chromossome as Chromo
import random
import math
import time

# Passo a passo


# Imprimir relatório - OK
# Implementar 3 memeticos, o meu, o com vnd, o com vns e o com vns e vnd // TODO:
# Implementar os algoritmos de Grasp - ...
# Buscar o ótimo das instancias atuais - ...
# Implementar as mesmas instancias utilizadas pelo outro algoritmo - OK
# Descobrir por que o meu ta demorando mais - ?
# Utilizar o IRACE para descobrir os melhores valores para as taxas nos algoritmos (mutation_rate, recombination_rate)
# Relatorio de todos algorítmos - OK...
# Teste estatístico e algoritmos de comparação
# Estado da arte + Modelagem do problema // TODO:
# IRACE -

# 2 Etapas
#   2.1. comparar os algoritmos memeticos entre si e os grasps entre si e reportar os experimentos no trabalho para encontrar o melhor grasp e o melhor memetico
#   2.2. comparar os resultados dos algritmos elegidos com o algoritmo da literatura
#   2.3. comparar os resultados dos algritmos elegidos com o ótimos das instancias

# investigar o solver do python e o algoritmo q ele usa //
#   Mixed integer Programming Solver using Coin CBC.


# 1. Introdução:
# 1.2 Contextualização e Motivação
# 1.3 Objetivos
# Gerais e especificos
# 1.4 Metodologia de Pesquisa
# 1.5 Contribuições desse trabalho para a literatura
# 1.6 Estrutura do Trabalho
# 2. Definição Formal do Problema
# 3


# teste estatistico
# teste T
# teste de mann-whitney U


# 1 Introdução
# 2 Definição formal do problema - FAZER
# 3 O estado da arte - FAZER - Um artigo por parágrafo
# 4 O algoritmo desenvolvido - FAZER
# 5 Experimentos
#   5.1 Metodologia
#       5.1.1 Descrever as características das instâncias
#   5.2 Resultados
#       5.21 Resultados para instâncias simétricas
#       5.22 Resultados para instâncias assimétricas
# 6 Conclusão | Considerações
# 7 Referências Bibliográficas


# 5 Experimentos
# 		5.1 Metodologia
# 			Características das instâncias, dos testes estatísitocos, quais análises serão feitas, parâmetros, e algoritmos comparativos.
# 		5.2 Resultaods
# 				5.2.1 Comparação dos algoritmos entre si
# 						Comparar os algoritmos propostos entre si, todos com todos, utilizando o teste de Kruskal-Wallis. Eleger o melhor grasp e o melhor memético. Utilizar instâncias simétricas e assimétricas.
# 5.2.2 Comparação com a literatura.
# 	Comparar os dois melhores algoritmos com os do solver


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


# enquanto critério de parada não for satisfeito faça:
# para cada i de 1 até N (onde N é o tamanho da população) faça:
# sorteie os pais. Aplique operadores (recombination, mutation).
# busca local do resultante da mutação
# busca local de alguns filhos gerados
# combinar populações de pais e filhos
# retornar o melhor indivíduo

# usar GUROBI


# Lentidão
# There are a few other things you can try:

# Profile your code to identify bottlenecks: Use a profiling tool to identify the parts of your code that take the most time. Once you know where the problem lies, you can try to optimize that specific part.

# Use Cython: Cython is a superset of Python that allows you to write C extensions for Python. By converting your Python code to Cython, you can achieve significant speedups.

# Use Numba: Numba is a just-in-time compiler that allows you to speed up Python code by compiling it on the fly. It works by generating optimized machine code for your functions, and can be a good option for numerical computations.

# ALGORITTMOS

population_size = 100
mutation_rate = 0.1
recombination_rate = 0.1

million = 1000000


def memetic_algorithm(G, quota):
    population = init_population(G, quota)

    # enquanto critério de parada não for satisfeito faça:
    while (route_cost.counter < million):
        children = []

        # para cada i de 1 até N (onde N é o tamanho da população) faça:
        for i in range(population_size):

            # sorteie os pais. Aplique operadores (recombination, mutation).
            parent_1 = parents_selection(population, 4)
            parent_2 = parents_selection(population, 4)

            # aplicar recombinação
            rand = random.random()
            if (rand < recombination_rate):
                child_1, child_2 = recombination(parent_1, parent_2, G)
                children.append(child_1)
                children.append(child_2)

            # aplicar mutação
            if (rand < mutation_rate):
                mutated = mutation(population, G)

                # busca local do resultante da mutação
                new_mutated = local_search(mutated, quota, G)
                count_local_mutation += 1

                children.append(new_mutated)
            # break

        # busca local de alguns filhos gerados
        random_children = random.choices(population, k=3)
        new_child_1 = local_search(random_children[0], quota, G)
        count_local += 1

        random_children = random.choices(population, k=3)
        new_child_2 = local_search(random_children[1], quota, G)
        count_local += 1

        random_children = random.choices(population, k=3)
        new_child_3 = local_search(random_children[2], quota, G)
        count_local += 1

        children.append(new_child_1)
        children.append(new_child_2)
        children.append(new_child_3)

        new_population = [*population, *children]

        new_population.sort(
            key=lambda item: item.fitness_value(), reverse=True)
        population = new_population[:population_size]
        # break

    # buscar melhor solução da população
    best = best_genetic_solution(population)
    route_cost.counter = 0

    return best


def vns_vnd_memetic_algorithm(G, quota):
    population = init_population(G, quota)

    # enquanto critério de parada não for satisfeito faça:
    while (route_cost.counter < million):

        children = []
        # para cada i de 1 até N (onde N é o tamanho da população) faça:
        for i in range(population_size):

            # sorteie os pais. Aplique operadores (recombination, mutation).
            parent_1 = parents_selection(population, 4)
            parent_2 = parents_selection(population, 4)

            # aplicar recombinação
            rand = random.random()
            if (rand < recombination_rate):
                child_1, child_2 = recombination(parent_1, parent_2, G)
                children.append(child_1)
                children.append(child_2)

            # aplicar mutação
            rand = random.random()
            if (rand < mutation_rate):
                mutated = mutation(population, G)
                # busca local do resultante da mutação
                new_mutated_route = vns_vnd(mutated.route, quota, G)
                new_mutated = Chromo(new_mutated_route, G)
                children.append(new_mutated)

        # busca local de alguns filhos gerados
        random_children = random.choices(population, k=3)
        new_child_1_route = vns_vnd(random_children[0].route, quota, G)

        new_child_2_route = vns_vnd(random_children[1].route, quota, G)

        new_child_3_route = vns_vnd(random_children[2].route, quota, G)

        new_child_1 = Chromo(new_child_1_route, G)
        new_child_2 = Chromo(new_child_2_route, G)
        new_child_3 = Chromo(new_child_3_route, G)

        children.append(new_child_1)
        children.append(new_child_2)
        children.append(new_child_3)

        new_population = [*population, *children]

        new_population.sort(
            key=lambda item: item.fitness_value(), reverse=True)

        population = new_population[:population_size]

    # buscar melhor solução da população
    best = best_genetic_solution(population)
    route_cost.counter = 0

    return best


def vns_memetic_algorithm(G, quota):
    population = init_population(G, quota)

    # enquanto critério de parada não for satisfeito faça:
    while (route_cost.counter < million):
        children = []
        # print('route_cost.counter', route_cost.counter)

        # para cada i de 1 até N (onde N é o tamanho da população) faça:
        for i in range(population_size):

            # sorteie os pais. Aplique operadores (recombination, mutation).
            parent_1 = parents_selection(population, 4)
            parent_2 = parents_selection(population, 4)

            # aplicar recombinação
            rand = random.random()
            if (rand < recombination_rate):
                child_1, child_2 = recombination(parent_1, parent_2, G)
                children.append(child_1)
                children.append(child_2)

            # aplicar mutação
            rand = random.random()
            if (rand < mutation_rate):
                mutated = mutation(population, G)
                # busca local do resultante da mutação
                new_mutated_route = vns(mutated.route, quota, G)
                new_mutated = Chromo(new_mutated_route, G)
                children.append(new_mutated)

        # busca local de alguns filhos gerados
        random_children = random.choices(population, k=3)
        new_child_1_route = vns(random_children[0].route, quota, G)

        new_child_2_route = vns(random_children[1].route, quota, G)

        new_child_3_route = vns(random_children[2].route, quota, G)

        new_child_1 = Chromo(new_child_1_route, G)
        new_child_2 = Chromo(new_child_2_route, G)
        new_child_3 = Chromo(new_child_3_route, G)

        children.append(new_child_1)
        children.append(new_child_2)
        children.append(new_child_3)

        new_population = [*population, *children]

        new_population.sort(
            key=lambda item: item.fitness_value(), reverse=True)

        population = new_population[:population_size]

    # buscar melhor solução da população
    best = best_genetic_solution(population)
    route_cost.counter = 0

    return best


def vnd_memetic_algorithm(G, quota):
    population = init_population(G, quota)

    # enquanto critério de parada não for satisfeito faça:
    while (route_cost.counter < million):

        children = []
        # para cada i de 1 até N (onde N é o tamanho da população) faça:
        for i in range(population_size):

            # sorteie os pais. Aplique operadores (recombination, mutation).
            parent_1 = parents_selection(population, 4)
            parent_2 = parents_selection(population, 4)

            # aplicar recombinação
            rand = random.random()
            if (rand < recombination_rate):
                child_1, child_2 = recombination(parent_1, parent_2, G)
                children.append(child_1)
                children.append(child_2)

            # aplicar mutação
            rand = random.random()
            if (rand < mutation_rate):
                mutated = mutation(population, G)
                # busca local do resultante da mutação
                new_mutated_route = vnd(mutated.route, quota, G)
                new_mutated = Chromo(new_mutated_route, G)

                children.append(new_mutated)

        # busca local de alguns filhos gerados
        random_children = random.choices(population, k=3)
        new_child_1_route = vnd(random_children[0].route, quota, G)

        new_child_2_route = vnd(random_children[1].route, quota, G)

        new_child_3_route = vnd(random_children[2].route, quota, G)

        new_child_1 = Chromo(new_child_1_route, G)
        new_child_2 = Chromo(new_child_2_route, G)
        new_child_3 = Chromo(new_child_3_route, G)

        children.append(new_child_1)
        children.append(new_child_2)
        children.append(new_child_3)

        new_population = [*population, *children]

        new_population.sort(
            key=lambda item: item.fitness_value(), reverse=True)

        population = new_population[:population_size]

    # buscar melhor solução da população
    best = best_genetic_solution(population)
    route_cost.counter = 0

    return best


def genetic_algorithm(G, quota):
    population = init_population(G, quota)

    # enquanto critério de parada não for satisfeito faça:
    while (route_cost.counter < million):

        children = []
        # para cada i de 1 até N (onde N é o tamanho da população) faça:
        for i in range(population_size):

            # sorteie os pais. Aplique operadores (recombination, mutation).
            parent_1 = parents_selection(population, 4)
            parent_2 = parents_selection(population, 4)

            # aplicar recombinação
            rand = random.random()
            if (rand < recombination_rate):
                child_1, child_2 = recombination(parent_1, parent_2, G)
                children.append(child_1)
                children.append(child_2)

            # aplicar mutação
            rand = random.random()
            if (rand < mutation_rate):
                mutated = mutation(population, G)
                children.append(mutated)

        new_population = [*population, *children]
        new_population.sort(
            key=lambda item: item.fitness_value(), reverse=True)
        population = new_population[:population_size]

    # buscar melhor solução da população
    best = best_genetic_solution(population)
    route_cost.counter = 0

    return best


def grasp_algorithm(G, quota):
    grasp_solutions = []
    alfa_grasp = 0.2
    best_cost = math.inf

    counter = 0

    while (route_cost.counter < million):

        # tic_construction = time.perf_counter()
        route = grasp_construction(G, quota, alfa_grasp)
        # tac_construction = time.perf_counter()
        # print('construction - ', tac_construction - tic_construction)

        grasp_solutions.append(route)

        # tic_drop = time.perf_counter()
        route = ls.drop_step(route, quota, G)
        # tac_drop = time.perf_counter()
        # print('drop - ', tac_drop - tic_drop)

        # tic_swap = time.perf_counter()
        route = ls.swap_2_opt(route, quota, G)
        # tac_swap = time.perf_counter()
        # print('swap - ', tac_swap - tic_swap)

        cost = route_cost(route, G)

        if (cost < best_cost):
            best_route = route
            best_cost = cost

    route_cost.counter = 0
    return best_route


def reactive_grasp_algorithm(G, quota):
    best_cost = math.inf
    alfas = [0.1, 0.2, 0.3, 0.4]
    weights = [1, 1, 1, 1]
    grasp_solutions = []
    alfa_costs = {alfa: [] for alfa in alfas}

    while (route_cost.counter < million):
        iterations = 1000
        k_itr = 100
        k = k_itr

        # print('route_cost.counter')
        # print(route_cost.counter)

        # print('weights')
        # print(weights)

        for i in range(iterations):
            alfa_grasp = random.choices(alfas, weights=weights, k=1)[0]

            route = grasp_construction(G, quota, alfa_grasp)

            if (route in grasp_solutions):
                continue

            grasp_solutions.append(route)

            route = ls.drop_step(route, quota, G)

            route = ls.swap_2_opt(route, quota, G)

            cost = route_cost(route, G)

            alfa_costs[alfa_grasp].append(cost)

            if cost < best_cost:
                best_route = route
                best_cost = cost

            if i == k:
                alfa_avgs = {alfa: sum(costs) / len(costs)
                             for alfa, costs in alfa_costs.items()}

                q_total = sum(best_cost / alfa_avgs[alfa] for alfa in alfas)
                weights = [best_cost / (q_total * alfa_avgs[alfa])
                           for alfa in alfas]

                k += k_itr

    route_cost.counter = 0
    return best_route


# def adaptative_grasp_algorithm(G, quota):
#     best_cost = math.inf
#     alfas = [0.1, 0.2, 0.3, 0.4]
#     weights = [1, 1, 1, 1]

#     while (route_cost.counter < million):
#         iterations = 5000
#         k_itr = 100
#         k = k_itr

#         alfa1 = []
#         alfa2 = []
#         alfa3 = []
#         alfa4 = []

#         grasp_solutions = []

#         print('route_cost.counter')
#         print(route_cost.counter)

#         print('weights')
#         print(weights)

#         for i in range(iterations):
#             alfa_grasp = random.choices(alfas, weights=weights, k=1)[0]

#             route = grasp_construction(G, quota, alfa_grasp)

#             if (route in grasp_solutions):
#                 continue

#             grasp_solutions.append(route)

#             route = ls.drop_step(route, quota, G)

#             route = ls.swap_2_opt(route, quota, G)

#             cost = route_cost(route, G)

#             if (alfa_grasp == alfas[0]):
#                 alfa1.append(cost)
#             if (alfa_grasp == alfas[1]):
#                 alfa2.append(cost)
#             if (alfa_grasp == alfas[2]):
#                 alfa3.append(cost)
#             if (alfa_grasp == alfas[3]):
#                 alfa4.append(cost)

#             if (cost < best_cost):
#                 best_route = route
#                 best_cost = cost

#             if (i == k):
#                 alfa1_av = sum(alfa1) / len(alfa1)
#                 alfa2_av = sum(alfa2) / len(alfa2)
#                 alfa3_av = sum(alfa3) / len(alfa3)
#                 alfa4_av = sum(alfa4) / len(alfa4)

#                 q1 = best_cost/alfa1_av
#                 q2 = best_cost/alfa2_av
#                 q3 = best_cost/alfa3_av
#                 q4 = best_cost/alfa4_av
#                 q_total = q1+q2+q3+q4
#                 weights = [q1/q_total, q2/q_total, q3/q_total, q4/q_total]
#                 k += k_itr

#     route_cost.counter = 0
#     return best_route


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
    len_min = begin = min(len(parent_1.route), len(parent_2.route))
    point_1, point_2 = random.sample(range(1, len_min-1), 2)
    begin = min(point_1, point_2)
    end = max(point_1, point_2)

    child_1 = parent_1.route[begin:end]
    child_2 = parent_2.route[begin:end]

    child_1_remain = [
        item for item in parent_2.route[1:] if item not in child_1]
    child_2_remain = [
        item for item in parent_1.route[1:] if item not in child_2]

    child_1 += child_1_remain
    child_2 += child_2_remain

    child_1.insert(0, parent_1.route[0])

    child_2.insert(0, parent_2.route[0])

    return child_1, child_2


def mutation(population, G):
    # TODO: adicionartaxa de mutação
    choosen = random.choice(population)

    first, second = random.sample(range(1, len(choosen.route)), 2)

    i = min(first, second)
    j = max(first, second)

    new_route = swap_2(i, j, choosen.route)
    new_chormo = Chromo(new_route, G)
    return new_chormo


def local_search(chromo, quota, G):
    # tic_swap = time.perf_counter()
    new_route = ls.swap_2_opt(chromo.route, quota, G)
    # tac_swap = time.perf_counter()
    # print('swap - ', tac_swap - tic_swap)

    new_chromo = Chromo(new_route, G)
    return new_chromo


def best_genetic_solution(population):
    best_chromo = population[0]
    for i in range(population_size):
        fitness_value = population[i].fitness_value()
        if (fitness_value > best_chromo.fitness_value()):
            best_chromo = population[i]
    return best_chromo.route


# def generate_random_route(G, quota):
#     bonus_colected = 0
#     route = [G.nodes[0]]
#     while (bonus_colected < quota):
#         random_i = random.randrange(0, len(G.nodes))
#         if (G.nodes[random_i] not in route):
#             bonus_colected += G.nodes[random_i]['bonus']
#             route.append(G.nodes[random_i])

#     return route

def generate_random_route(G, quota):
    bonus_colected = 0
    route = [0]
    while (bonus_colected < quota):
        random_i = random.randrange(0, len(G.nodes))
        if (random_i not in route):
            bonus_colected += G.nodes[random_i]['bonus']
            route.append(random_i)

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
            i = route[r-1]
            j = route[r]
            s = route[r+1]
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
#                 if (len(route) == 1):
#                     route.insert(1, G.nodes[k])
#                     continue
#                 for r in range(len(route)):
#                     i = route[r-1]
#                     j = route[r]
#                     edge = G.edges[i, j]
#                     k_edge1 = G.edges[i, k]
#                     k_edge2 = G.edges[k, j]
#                     k_economy_value = edge['weight'] + G.nodes[k]['penalty'] - \
#                         k_edge1['weight'] - k_edge2['weight']
#                     economy_list.append((k, k_economy_value, r))
#         if (len(economy_list) == 0):
#             continue
#         economy_list.sort(key=lambda item: item[1], reverse=True)
#         best_economy = economy_list[0][1]
#         worst_economy = economy_list[-1][1]
#         grasp_tsh = best_economy - alfa_grasp * (best_economy - worst_economy)
#         grasp_candidates = list(
#             filter(lambda item: item[1] >= grasp_tsh, economy_list))
#         insertion_selected = random.choice(grasp_candidates)
#         if (best_economy > 0 or bonus_colected < quota):
#             route.insert(insertion_selected[2], G.nodes[insertion_selected[0]])
#         bonus_colected = calculate_bonus_colected(route, G)
#     return route


def grasp_construction(G, quota, alfa_grasp):

    nodes = list(filter(lambda item: item != 0, G.nodes))

    random_node = random.choice(nodes)

    route = [G.nodes[0], G.nodes[random_node]]

    bonus_colected = calculate_bonus_colected(route, G)
    best_economy = -math.inf

    while bonus_colected < quota or best_economy > 0:
        best_economy = -math.inf
        economy_list = []
        for k in range(len(G.nodes)):
            if G.nodes[k] not in route:
                i = route[-2]
                j = route[-1]
                edge = G.edges[i, j]
                k_edge1 = G.edges[i, k]
                k_edge2 = G.edges[k, j]
                k_economy_value = edge['weight'] + G.nodes[k]['penalty'] - \
                    k_edge1['weight'] - k_edge2['weight']
                economy_list.append((k, k_economy_value))

        if (len(economy_list) == 0):
            continue
        economy_list.sort(key=lambda item: item[1], reverse=True)
        best_economy = economy_list[0][1]
        worst_economy = economy_list[-1][1]
        grasp_tsh = best_economy - alfa_grasp * (best_economy - worst_economy)
        grasp_candidates = list(
            filter(lambda item: item[1] >= grasp_tsh, economy_list))
        insertion_selected = random.choice(grasp_candidates)
        if (best_economy > 0 or bonus_colected < quota):
            route.insert(len(route) - 1, G.nodes[insertion_selected[0]])
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

# Algoritmo GRASP + VNS + VND.
def grasp_vns_vnd(G, quota):
    max_iter = 4000
    alfa_grasp = 0.2

    # Construção
    route = [G.nodes[0]]
    best_route = route
    best_cost = math.inf
    bonus_colected = calculate_bonus_colected(route, G)

    for n in range(max_iter):
        best_economy = -math.inf
        economy_list = []
        for k in range(len(G.nodes)):
            # print('economy_list')
            # print(economy_list)
            if G.nodes[k] not in route:
                route_len = len(route)
                if (route_len == 1):
                    i = route[0]
                    edge = G.edges[0, k]
                    k_economy_value = G.nodes[k]['penalty'] - edge['weight']
                else:
                    i = route[-2]
                    j = route[-1]
                    edge = G.edges[i, j]
                    k_edge1 = G.edges[i, k]
                    k_edge2 = G.edges[k, j]
                    k_economy_value = edge['weight'] + G.nodes[k]['penalty'] - \
                        k_edge1['weight'] - k_edge2['weight']
                economy_list.append((k, k_economy_value))

        has_positive_economy = any(economy[1] > 0 for economy in economy_list)

        while (bonus_colected < quota or has_positive_economy):
            economy_list.sort(key=lambda item: item[1], reverse=True)
            best_economy = economy_list[0][1]
            worst_economy = economy_list[-1][1]
            grasp_tsh = best_economy - alfa_grasp * \
                (best_economy - worst_economy)
            grasp_candidates = list(
                filter(lambda item: item[1] >= grasp_tsh, economy_list))
            insertion_selected = random.choice(grasp_candidates)
            if (best_economy > 0 or bonus_colected < quota):
                route.append(G.nodes[insertion_selected[0]])
                economy_list.remove(insertion_selected)
            # TODO: Analisar se precisa recaul
            bonus_colected = calculate_bonus_colected(route, G)
            has_positive_economy = any(
                economy[1] > 0 for economy in economy_list)

        cost = route_cost(route, G)
        if (cost < best_cost):
            best_cost = cost
            best_route = route
    route = best_route

    print('route pre vndvns', best_cost)

    # VNS + VND
    best_route = vns_vnd(route, quota, G)

    print('route pos vndvns', route_cost(best_route, G))

    return best_route


def vns(route, quota, G):
    max_time = 200

    best_route = route
    best_cost = route_cost(best_route, G)

    no_improvement_time = 0
    neighboors = [neighboor_1,
                  neighboor_2,
                  neighboor_3,
                  neighboor_4,
                  neighboor_5]
    neighboors_len = len(neighboors)

    while (no_improvement_time < max_time):
        init_time = time.perf_counter()
        k = 0

        while (k < neighboors_len):
            choosen_function = neighboors[k]

            neighboor = choosen_function(best_route, G)

            bonus_colected = calculate_bonus_colected(neighboor, G)

            if (bonus_colected >= quota):
                # local search
                new_route = ls.swap_2_opt(neighboor, quota, G)
                new_cost = route_cost(new_route, G)

                if (new_cost < best_cost):
                    best_cost = new_cost
                    best_route = new_route
                    k = 0
                else:
                    k += 1
            else:
                k += 1

            current_time = time.perf_counter()
            no_improvement_time = current_time - init_time

        return best_route
    return best_route


def vns_vnd(route, quota, G):
    max_time = 200

    best_route = route
    best_cost = route_cost(best_route, G)

    no_improvement_time = 0
    neighboors = [neighboor_1,
                  neighboor_2,
                  neighboor_3,
                  neighboor_4,
                  neighboor_5]
    neighboors_len = len(neighboors)

    while (no_improvement_time < max_time):
        init_time = time.perf_counter()
        k = 0

        while (k < neighboors_len):
            choosen_function = neighboors[k]

            neighboor = choosen_function(best_route, G)

            bonus_colected = calculate_bonus_colected(neighboor, G)

            # VND
            vnd_route = vnd(neighboor, quota, G)

            vnd_cost = route_cost(vnd_route, G)

            if (vnd_cost < best_cost):
                best_cost = vnd_cost
                best_route = vnd_route
                k = 0
            else:
                k += 1

            current_time = time.perf_counter()
            no_improvement_time = current_time - init_time
        return best_route
    return best_route


def vnd(route, quota, G):
    max_time = 200

    best_route = route
    best_cost = route_cost(best_route, G)

    no_improvement_time = 0
    refinements = [ls.seq_drop_seq_add,
                   ls.swap_2_opt,
                   ls.add_drop]
    refinements_len = len(refinements)

    while (no_improvement_time < max_time):
        init_time = time.perf_counter()
        k = 0

        while (k < refinements_len):
            choosen_function = refinements[k]

            refined_route = choosen_function(route, quota, G)

            refined_cost = route_cost(refined_route, G)

            if (refined_cost < best_cost):
                best_cost = refined_cost
                best_route = refined_route
                k = 0
            else:
                k += 1

            current_time = time.perf_counter()
            no_improvement_time = current_time - init_time

        return best_route

    return route

# Remove vértice com maior economia:


def neighboor_1(route, G):
    # print('neighboor_1')
    economy_list = []
    for r in range(len(route) - 1):
        i = route[r-1]
        j = route[r]
        s = route[r+1]
        k_edge1 = G.edges[i, j]
        k_edge2 = G.edges[j, s]
        edge = G.edges[i, s]
        k_economy_value = k_edge1['weight'] + k_edge2['weight'] - \
            edge['weight'] - G.nodes[j]['penalty']

        economy_list.append((j, k_economy_value))

    economy_list.sort(key=lambda item: item[1], reverse=True)
    best_economy_item = economy_list[0]
    route.remove(G.nodes[best_economy_item[0]])
    # print(route)
    return route

# Inserir vértice de maior economia:


def neighboor_2(route, G):
    # print('neighboor_2')

    best_economy = -math.inf

    # insert
    best_economy = -math.inf
    economy_list = []
    for k in range(len(G.nodes)):
        if G.nodes[k] not in route:
            for r in range(1, len(route)):
                i = route[r-1]
                j = route[r]
                edge = G.edges[i, j]
                k_edge1 = G.edges[i, k]
                k_edge2 = G.edges[k, j]
                k_economy_value = edge['weight'] + G.nodes[k]['penalty'] - \
                    k_edge1['weight'] - k_edge2['weight']
                economy_list.append((k, k_economy_value, r))
    economy_list.sort(key=lambda item: item[1], reverse=True)
    best_economy = economy_list[0]
    if (best_economy[2] == 0):
        route.append(G.nodes[best_economy[0]])
    else:
        route.insert(best_economy[2], G.nodes[best_economy[0]])

    # print(route)
    return route


# Troca dois vertices:


def neighboor_3(route, G):
    # print('neighboor_3')
    first, second = random.sample(range(1, len(route)), 2)

    i = min(first, second)
    j = max(first, second)
    route = swap_2(i, j, route)
    # print(route)
    return route

# Remove vertice aleatorio:


def neighboor_4(route, G):
    # print('neighboor_4')
    random_i = random.randrange(1, len(route))

    del route[random_i]

    # print(route)
    return route

# Insere vertice aleatorio:


def neighboor_5(route, G):
    # print('neighboor_5')
    random_node = random.choice(G.nodes)
    while (random_node in route):
        random_node = random.choice(G.nodes)
    random_i = random.randrange(1, len(route))

    route.insert(random_i, random_node)
    # print(route)
    return route
