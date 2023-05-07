#include <stdio.h>
#include <stdlib.h>

// graph.h

#ifndef GRAPH_H
#define GRAPH_H

// Graph Class

struct Graph
{
    int size;
    struct Node *nodes;
    struct Edge *edges;
};

struct Node
{
    int id;
    int bonus;
    int penalty;
};

struct Edge
{
    int node1;
    int node2;
    int weight;
};

struct Graph *create_graph(int size, int distance_matrix[][size], int node_data[][3])
{
    struct Graph *graph = malloc(sizeof(struct Graph));
    graph->size = size;
    graph->nodes = malloc(size * sizeof(struct Node));
    graph->edges = malloc(size * size * sizeof(struct Edge));

    // create nodes
    for (int i = 0; i < size; i++)
    {
        struct Node node = {i, node_data[i][1], node_data[i][2]};
        graph->nodes[i] = node;
    }

    // create edges
    int edge_count = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int weight = distance_matrix[i][j];
            if (weight > 0)
            {
                struct Edge edge = {i, j, weight};
                graph->edges[edge_count++] = edge;
            }
            else
            {
                struct Edge edge = {i, j, graph->nodes[i].penalty};
                graph->edges[edge_count++] = edge;
            }
        }
    }

    return graph;
}

void print_graph(struct Graph *graph)
{
    printf("Nodes:\n");
    for (int i = 0; i < graph->size; i++)
    {
        printf("%d (bonus: %d, penalty: %d)\n", graph->nodes[i].id, graph->nodes[i].bonus, graph->nodes[i].penalty);
    }

    printf("Edges:\n");
    for (int i = 0; i < graph->size * graph->size; i++)
    {
        printf("%d ---%d---> %d\n", graph->edges[i].node1, graph->edges[i].weight, graph->edges[i].node2);
    }
}

#endif // GRAPH_H
