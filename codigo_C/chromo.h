#include <stdio.h>
#include <stdlib.h>
#include "fitness.h"
#include "graph.h"

#ifndef CHROMO_H
#define CHROMO_H

struct Chromo
{
    int *route;
    int fitness;
};

struct Chromo create_chromo(int *route, struct Graph *graph)
{
    struct Chromo chromo;
    chromo.route = route;
    chromo.fitness = calculate_fitness(route, graph);
    return chromo;
}

void destroy_chromo(struct Chromo *chromo)
{
    free(chromo->route);
    free(chromo);
}

#endif