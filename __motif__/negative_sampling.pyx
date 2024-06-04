from cython import boundscheck, wraparound, cdivision, nonecheck
from libc.stdlib cimport malloc

from numpy.math cimport INFINITY

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
cdef double* floyd_warshall(double* A, int num_nodes) nogil:
    cdef:
        int k, i, j
        double *dist
    
    # Inizializza la matrice delle distanze con i pesi iniziali
    dist = <double*>malloc(num_nodes * num_nodes * sizeof(double))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist[i * num_nodes + j] = A[i * num_nodes + j]
    
    # Itera su tutti i nodi come vertici intermedi
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                # Aggiorna la distanza se un percorso più breve è trovato
                dist[i * num_nodes + j] = min(dist[i * num_nodes + j], dist[i * num_nodes + k] + dist[k * num_nodes + j])
    return dist

cdef class NegativeSampler:

    cdef double alpha
    cdef int beta
    
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @nonecheck(False)
    def __cinit__(self, short[:, :] incidence_matrix, double alpha, int beta, int[:, :] motif = None):
        cdef:
            double* A
            double* node_distances
            double* edge_distances
            int num_nodes, num_edges
            int i, j, k
        self.alpha = alpha
        self.beta = beta
        num_nodes = incidence_matrix.shape[0]
        num_edges = incidence_matrix.shape[1]
        A = <double*>malloc(num_nodes * num_nodes * sizeof(double))
        for i in range(num_nodes):
            for j in range(num_nodes):
                for k in range(num_edges):
                    A[i * num_nodes + j] += incidence_matrix[i, k] * incidence_matrix[j, k]
        node_distances = floyd_warshall(A, num_nodes)
        # The edge_distances are the distances between the edges and nodes
        edge_distances = <double*>malloc(num_edges * num_edges * sizeof(double))
        for i in range(num_edges):
                for j in range(num_nodes):
                    edge_distances[i * num_edges + j] = INFINITY
                    for k in range(num_nodes):
                        edge_distances[i * num_edges + j] = min(edge_distances[i * num_edges + j], node_distances[j * num_nodes + k])

    def sample(self):
        pass

