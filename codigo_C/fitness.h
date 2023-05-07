#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "globals.h"
#include "aux_functions.h"

#ifndef FITNESS_H
#define FITNESS_H

int route_cost_calls = 0;

int route_cost(int *route, struct Graph *graph)
{
    route_cost_calls++;
    int route_len = get_route_length(route);
    int cost = 0;
    int visited_nodes[graph->size];
    memset(visited_nodes, 0, sizeof(visited_nodes)); // initialize visited_nodes array to 0

    // Calculate the sum of the weights of the edges in the route
    for (int i = 0; i < route_len - 1; i++)
    {
        int node1 = route[i];
        int node2 = route[i + 1];
        for (int j = 0; j < graph->size * graph->size; j++)
        {
            if ((graph->edges[j].node1 == node1 && graph->edges[j].node2 == node2) || (graph->edges[j].node1 == node2 && graph->edges[j].node2 == node1))
            {
                cost += graph->edges[j].weight;
                break;
            }
        }
        visited_nodes[node1] = 1;
    }
    // Add the edge from the last node to the first node
    int last_node = route[route_len - 1];
    int first_node = route[0];
    for (int j = 0; j < graph->size * graph->size; j++)
    {
        if ((graph->edges[j].node1 == last_node && graph->edges[j].node2 == first_node) || (graph->edges[j].node1 == first_node && graph->edges[j].node2 == last_node))
        {
            cost += graph->edges[j].weight;
            break;
        }
    }
    visited_nodes[last_node] = 1;

    // Calculate the sum of penalties for the unvisited nodes
    for (int i = 0; i < graph->size; i++)
    {
        if (visited_nodes[i] == 0)
        {
            cost += graph->nodes[i].penalty;
        }
    }

    return cost;
}

int calculate_fitness(int *route, struct Graph *graph)
{
    int cost = route_cost(route, graph);
    return 1 / (cost + 1);
}

#endif