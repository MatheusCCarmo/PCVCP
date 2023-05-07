cimport numpy as np

cdef class Graph:
    cdef dict nodes
    cdef dict edges

    def __init__(self, np.ndarray[np.double_t, ndim=2] distance_matrix):
        self.nodes = {}
        self.edges = {}

        cdef int n = distance_matrix.shape[0]
        cdef int i, j
        cdef double weight

        for i in range(n):
            # default values for bonus and penalty attributes
            node = {"bonus": 0, "penalty": 0}
            self.add_node(i, node)

        for i in range(n):
            for j in range(n):
                weight = distance_matrix[i, j]
                self.add_edge(i, j, weight)

    cpdef add_node(self, int node_id, dict node):
        self.nodes[node_id] = node

    cpdef add_edge(self, int node1, int node2, double weight):
        self.edges[(node1, node2)] = {'weight': weight}

    def __str__(self):
        cdef str node_str, edge_str
        node_str = "Vertices:\n" + \
            "\n".join(str(node) + f" ({attrs})" for node,
                      attrs in self.nodes.items()) + "\n"
        edge_str = "Edges:\n" + \
            "\n".join(
                f"{str(edge[0])} -> {str(edge[1])} (weight={weight})" for edge, weight in self.edges.items())
        return f"{node_str}{edge_str}"
