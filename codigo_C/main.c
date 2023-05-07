
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "globals.h"
#include "graph.h"
#include "fitness.h"
#include "aux_functions.h"
#include "algorithms.c"

#define MAX_COST_CALLS 1000000

int main()
{
    clock_t start_time, end_time;
    double time_taken;
    int cost;
    int distance_matrix[10][10] = {
        {0, 193, 139, 134, 231, 158, 152, 249, 196, 171},
        {193, 0, 209, 136, 131, 227, 174, 149, 246, 240},
        {139, 209, 0, 205, 132, 127, 224, 170, 146, 242},
        {134, 136, 205, 0, 202, 149, 124, 220, 167, 142},
        {231, 131, 132, 202, 0, 198, 145, 120, 217, 163},
        {158, 227, 127, 149, 198, 0, 195, 142, 238, 213},
        {152, 174, 224, 124, 145, 195, 0, 192, 138, 235},
        {249, 149, 170, 220, 120, 142, 192, 0, 140, 135},
        {196, 246, 146, 167, 217, 238, 138, 140, 0, 136},
        {171, 240, 242, 142, 163, 213, 235, 135, 136, 0},
    };

    int node_data[10][3] = {
        {0, 0, 999},
        {1, 901, 50},
        {2, 367, 50},
        {3, 341, 50},
        {4, 726, 50},
        {5, 941, 50},
        {6, 916, 50},
        {7, 550, 50},
        {8, 525, 50},
        {9, 499, 50},
    };

    int quota = 3020;

    struct Graph *graph = create_graph(10, distance_matrix, node_data);
    int *route = generate_random_route(graph, quota);

    int route_len = get_route_length(route);

    printf("\nRoute length: %d\n", route_len);
    printf("\nROUTE: ");

    for (int i = 0; i < route_len; i++)
    {
        printf("%d -> ", route[i]);
    }

    start_time = clock(); // get the start time

    while (route_cost_calls < MAX_COST_CALLS)
    {
        cost = route_cost(route, graph);
    }

    end_time = clock(); // get the end time

    time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC; // calculate the time taken

    printf("\nTime taken: %f seconds\n", time_taken);

    printf("Route cost: %d\n", cost);

    return 0;
}
