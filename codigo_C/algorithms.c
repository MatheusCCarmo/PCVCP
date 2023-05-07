#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "graph.h"
#include "chromo.h"
#include "aux_functions.h"

int *generate_random_route(struct Graph *graph, int quota)
{
    srand(time(NULL));

    int bonus_colected = 0;
    int *route = malloc(graph->size * sizeof(int));
    int route_index = 0;
    route[route_index++] = 0; // start at node 0
    while (bonus_colected < quota && route_index < graph->size)
    {
        int random_i = rand() % graph->size;
        if (random_i != 0 && !contains(route, route_index, random_i))
        {
            bonus_colected += graph->nodes[random_i].bonus;
            route[route_index++] = random_i;
        }
    }
    return route;
}

struct Chromo *init_population(struct Graph *graph, int population_size, int quota)
{
    struct Chromo *population = malloc(population_size * sizeof(struct Chromo));
    for (int i = 0; i < population_size; i++)
    {
        int *route = generate_random_route(graph, quota);
        population[i] = create_chromo(route, graph);
    }
    return population;
}

// // Function to perform crossover between two parent chromosomes
// void crossover(int *parent1, int *parent2, int *child, int num_nodes)
// {
//     // code for performing crossover
// }

// // Function to perform mutation on an individual chromosome
// void mutate(int *chromosome, int num_nodes)
// {
//     // code for performing mutation
// }

// // Function to select parents for crossover using tournament selection
// void tournament_selection(int *population, int *parent1, int *parent2, int num_nodes, int population_size)
// {
//     // code for tournament selection
// }

// // Function to generate the next generation of the population
// void next_generation(int *population, struct Graph *graph, int num_nodes, int population_size)
// {
//     // code for generating the next generation of the population
// }
