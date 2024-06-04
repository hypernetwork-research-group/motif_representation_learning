# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound, nonecheck
from cython.parallel import prange, threadid

cnp.import_array()

ctypedef cnp.float32_t DTYPE_t
DTYPE = np.float32

ctypedef cnp.int32_t DTYPE_int_t
DTYPE_int = np.int32


@boundscheck(False)
@wraparound(False)
@nonecheck(False)
cdef floyd_warshall(cnp.ndarray[cnp.float64_t, ndim=2] incidence_matrix):
    cdef:
        int N, k, i, j
        cnp.ndarray[cnp.float64_t, ndim=2] dist

    # Numero di nodi nel grafo
    N = incidence_matrix.shape[0]
    
    # Inizializza la matrice delle distanze con i pesi iniziali
    dist = incidence_matrix.copy()
    
    # Itera su tutti i nodi come vertici intermedi
    for k in range(N):
        for i in range(N):
            for j in range(N):
                # Aggiorna la distanza se un percorso più breve è trovato
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dist

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
cdef compute_edge_distances(cnp.ndarray[cnp.int16_t, ndim=2] incidence_matrix):
    cdef:
        cnp.ndarray[cnp.float64_t, ndim=2] A
        cnp.ndarray[cnp.float64_t, ndim=2] node_distances
        cnp.ndarray[cnp.float64_t, ndim=2] edge_distances
        int n_edges, n_nodes, i, j, k
    n_edges = incidence_matrix.shape[1]
    n_nodes = incidence_matrix.shape[0]
    A = (incidence_matrix @ incidence_matrix.T).astype(np.float64)
    A = np.where(A, 1, np.inf)
    np.fill_diagonal(A, 0)
    node_distances = floyd_warshall(A)
    edge_distances = np.zeros((n_nodes, n_edges), dtype=np.float64).T
    for i, e in enumerate(incidence_matrix.T.astype(np.bool_)):
        edge_distances[i] = node_distances[e].min(axis=0)
    return node_distances, edge_distances

from time import time
cimport openmp

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def xmotif_negative_sampling(cnp.ndarray[cnp.int16_t, ndim=2] incidence_matrix, cnp.ndarray[DTYPE_int_t, ndim=2] motifs, float alpha, int beta):
    cdef:
        long[:] node_degrees
        double[:, :] node_distances, edge_distances
        int num_nodes = incidence_matrix.shape[0]
        int num_motifs = motifs.shape[0]
        int motif_size = motifs.shape[1]
        int i
        # For multithreading
        int num_threads = openmp.omp_get_max_threads()
        int _threadid

    node_degrees = incidence_matrix.sum(axis=1)
    node_distances, edge_distances = compute_edge_distances(incidence_matrix)

    for i in prange(num_motifs, nogil=True):
        _threadid = threadid()
        motif_size += 1

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def motif_negative_sampling(cnp.ndarray[cnp.int16_t, ndim=2] incidence_matrix, cnp.ndarray[DTYPE_int_t, ndim=2] motifs, float alpha, int beta):
    cdef:
        cnp.ndarray[cnp.int64_t, ndim=1] node_degrees
        cnp.ndarray[cnp.float64_t, ndim=2] node_distances, edge_distances
        cnp.ndarray[cnp.int16_t, ndim=2] motif_im
        cnp.ndarray[DTYPE_int_t, ndim=2] motifs_
        cnp.ndarray[DTYPE_int_t, ndim=1] motif_unique_nodes
        cnp.ndarray[cnp.float64_t, ndim=1] motif_distances
        cnp.ndarray[DTYPE_int_t, ndim=1] new_motif
        cnp.ndarray[DTYPE_t, ndim=2] y_e, y_m
        list new_motifs = [], new_edges = []
        int edge_i, num_nodes, num_motifs
        int m_i, b
    node_degrees = incidence_matrix.sum(axis=1)
    node_distances, edge_distances = compute_edge_distances(incidence_matrix)
    edge_i = incidence_matrix.shape[1] - 1
    num_nodes = incidence_matrix.shape[0]
    num_motifs = motifs.shape[0]
    for m_i in range(num_motifs):
        m = motifs[m_i]
        # From here to here
        motif_im = incidence_matrix[:, m].T
        motif_unique_nodes = (motif_im.sum(axis=0) > 0).nonzero()[0].astype(DTYPE_int)

        motif_distances = edge_distances[m].min(axis=0)
        motif_distances += 1
        motif_distances[motif_distances != np.inf] = motif_distances[motif_distances != np.inf].max() - motif_distances[motif_distances != np.inf] + 1 # Invert the distances
        motif_distances[motif_distances == np.inf] = 0 # Set the unreachable nodes score to 0
        p_dist = motif_distances / motif_distances.sum() # Compute the probability distribution, the lower the distance the higher the probability
        p_deg = node_degrees / node_degrees.sum()
        p = p_dist + p_deg
        p[motif_unique_nodes] = 0 # Exclude nodes already in the motif
        p /= p.sum()
        
        # To here, the execution time is around 150 ms

        for b in range(beta):
            # This instruction needs ~20 ms
            nodes_to_replace = motif_unique_nodes[np.random.rand(motif_unique_nodes.shape[0]) > alpha] # Randomly select nodes to replace

            # Execution time without this instruction is around 154 ms
            # This instruction needs ~200 ms
            # replaced_nodes = np.random.choice(num_nodes, nodes_to_replace.shape[0], replace=False, p=p) # Randomly select nodes to replace, using the probability distribution p (stocastic sampling)
            replaced_nodes = p.argsort()[::-1][:nodes_to_replace.shape[0]] # Select nodes using the order of p (greedy sampling)

            # # # This block needs ~120 ms - Execution time without this block is around 358 ms
            new_motif = np.zeros_like(m)
            for i, (_m, motif_index) in enumerate(zip(motif_im, m)):
                new_edge = np.copy(_m)
                for n, r in zip(nodes_to_replace, replaced_nodes):
                    if new_edge[n]:
                        new_edge[n] = 0
                        new_edge[r] = 1
                if not (_m == new_edge).all():
                    edge_i += 1
                    new_motif[i] = edge_i
                    new_edges.append(new_edge)
                else:
                    new_motif[i] = motif_index
            if (new_motif == m).all():
                continue
            new_motifs.append(new_motif)

    new_incidence_matrix = np.hstack((incidence_matrix, np.array(new_edges).T))
    motifs_ = np.vstack((motifs, np.array(new_motifs)), dtype=DTYPE_int)
    y_e = np.vstack((np.ones((incidence_matrix.shape[1], 1)), np.zeros((len(new_edges), 1))), dtype=DTYPE)
    y_m = np.vstack((np.ones((motifs.shape[0], 1)), np.zeros((len(new_motifs), 1))), dtype=DTYPE)

    new_incidence_matrix, unique_indices, unique_inverse = np.unique(new_incidence_matrix, axis=1, return_index=True, return_inverse=True)
    motifs_ = np.sort(unique_inverse[motifs_], axis=1).astype(DTYPE_int)
    y_e = y_e[unique_indices]

    return new_incidence_matrix, motifs_, y_e, y_m

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def expected_degree_hypergraph(node_degrees, edge_degrees, num_interactions):
    incidence_matrix = np.zeros((node_degrees.shape[0], edge_degrees.shape[0])).astype(DTYPE)
    node_degrees_ = node_degrees / node_degrees.sum()
    edge_degrees_ = edge_degrees / edge_degrees.sum()
    n = np.random.choice(incidence_matrix.shape[0], num_interactions, p=node_degrees_)
    e = np.random.choice(incidence_matrix.shape[1], num_interactions, p=edge_degrees_)
    incidence_matrix[n, e] = 1
    incidence_matrix = incidence_matrix[:, incidence_matrix.sum(axis=0) != 0]
    return incidence_matrix

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def node_node_interactions(short[:, :] X, int nodes_threshold = 1):
    # Parse all the test data to tensors and compute edge indexes
    # Edge indexes are used to compute the convolutional operations
    cdef:
        int num_nodes = X.shape[0]
        int num_edges = X.shape[1]
        int i, j
        int e, n, m
        int interactions, count
        #
        int[:] _interactions
        int num_threads = openmp.omp_get_max_threads()
        int _threadid
        # node-node interactions
        long[:, :, :] local_node_edge_index
        long[:, :] node_edge_index
    # Compute node-node interactions
    interactions = 0
    for i in prange(num_nodes, nogil=True):
        for j in range(i + 1, num_nodes):
            count = 0
            for e in range(num_edges):
                if X[i, e] and X[j, e]:
                    count = count + 1
                    if count >= nodes_threshold:
                        interactions += 2
                        break
    _interactions = np.zeros(num_threads, dtype=np.int32)
    local_node_edge_index = np.zeros((num_threads, 2, interactions), dtype=np.int64)
    for i in prange(num_nodes, nogil=True):
        _threadid = threadid()
        for j in range(i + 1, num_nodes):
            count = 0
            for e in range(num_edges):
                if X[i, e] and X[j, e]:
                    count = count + 1
                    if count >= nodes_threshold:
                        local_node_edge_index[_threadid, 0, _interactions[_threadid]] = i
                        local_node_edge_index[_threadid, 1, _interactions[_threadid]] = j
                        _interactions[_threadid] += 1
                        local_node_edge_index[_threadid, 0, _interactions[_threadid]] = j
                        local_node_edge_index[_threadid, 1, _interactions[_threadid]] = i
                        _interactions[_threadid] += 1
                        break
    node_edge_index = np.zeros((2, interactions), dtype=np.int64)
    interactions = 0
    for i in range(num_threads):
        node_edge_index[:, interactions:interactions + _interactions[i]] = local_node_edge_index[i, :, :_interactions[i]]
        interactions += _interactions[i]
    return np.array(node_edge_index)

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def edge_motif_interactions(short[:, :] X, int[:, :] motifs):
    # Parse all the test data to tensors and compute edge indexes
    # Edge indexes are used to compute the convolutional operations
    cdef:
        int num_motifs = motifs.shape[0]
        int motif_size = motifs.shape[1]
        int i
        long e, m
        int interactions
        # edge-motif interactions
        long[:, :] edge_motif_index
    interactions = 0
    edge_motif_index = np.zeros((2, num_motifs * motif_size), dtype=np.int64)
    for m in range(num_motifs):
        for i in range(motif_size):
            e = motifs[m, i]
            edge_motif_index[0, interactions] = e
            edge_motif_index[1, interactions] = m
            interactions += 1
    return np.array(edge_motif_index)
