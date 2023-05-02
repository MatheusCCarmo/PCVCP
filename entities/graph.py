import numpy as np


class Graph:
    def __init__(self, distance_matrix):
        self.nodes = {}
        self.edges = {}

        for i in range(distance_matrix.shape[0]):
            # default values for bonus and penalty attributes
            node = {"bonus": 0, "penalty": 0}
            self.add_node(i, node)

        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[0]):
                weight = distance_matrix[i, j]
                self.add_edge(i, j, weight)

    def add_node(self, node_id, node):
        self.nodes[node_id] = node

    def add_edge(self, node1, node2, weight):
        self.edges[(node1, node2)] = {'weight': weight}

    def __str__(self):
        node_str = "Vertices:\n" + \
            "\n".join(str(node) + f" ({attrs})" for node,
                      attrs in self.nodes.items()) + "\n"
        edge_str = "Edges:\n" + \
            "\n".join(
                f"{str(edge[0])} -> {str(edge[1])} (weight={weight})" for edge, weight in self.edges.items())
        return f"{node_str}{edge_str}"
