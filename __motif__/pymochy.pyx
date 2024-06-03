# cython: language_level=3
from cython import boundscheck, wraparound, cdivision, nonecheck
from libc.stdlib cimport malloc, free
from cython.parallel import prange, threadid
from libc.stdlib cimport rand, RAND_MAX
cimport openmp
import numpy as np
cimport numpy as cnp

@boundscheck(False)
@wraparound(False)
@cdivision(True)
@nonecheck(False)
cdef inline int min(int a, int b) nogil:
    if a < b:
        return a
    return b

@boundscheck(False)
@wraparound(False)
@cdivision(True)
@nonecheck(False)
cdef inline int max(int a, int b) nogil:
    if a > b:
        return a
    return b

@boundscheck(False)
@wraparound(False)
@cdivision(True)
@nonecheck(False)
cdef inline long get_common_neighbor(long* common_neighbors, int num_edges, int i, int j) noexcept nogil:
    cdef:
        int m, M
    m = min(i, j)
    M = max(i, j)
    return common_neighbors[m * num_edges + M - ((m * (m + 1))//2)]

@boundscheck(False)
@wraparound(False)
@cdivision(True)
@nonecheck(False)
cdef int get_motif_id(int deg_i, int deg_j, int deg_k, int c_ij, int c_ik, int c_jk, int g_ijk) nogil:
    cdef:
        int a, b, c, d, e, f, g
    a = deg_i - (c_ij + c_ik)   + g_ijk
    b = deg_j - (c_jk + c_ij) + g_ijk
    c = deg_k - (c_ik + c_jk) + g_ijk
    d = c_ij - g_ijk
    e = c_jk - g_ijk
    f = c_ik - g_ijk
    g = g_ijk
    return (a > 0) + ((b > 0) << 1) + ((c > 0) << 2) + ((d > 0) << 3) + ((e > 0) << 4) + ((f > 0) << 5) + ((g > 0) << 6)

cdef class CommonNeighbors:

    cdef short[:, :] incidence_matrix
    cdef long* common_neighbors
    cdef int num_edges
    cdef int optimized

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @nonecheck(False)
    def __cinit__(self, short[:, :] incidence_matrix, int num_edges, int optimized):
        self.incidence_matrix = incidence_matrix
        self.num_edges = num_edges
        self.optimized = optimized
        if self.optimized == 1:
            self.common_neighbors = CommonNeighbors.common_neighbors_matrix(incidence_matrix)

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @nonecheck(False)
    cdef inline long get(self, int i, int j) nogil:
        cdef:
            int m, M
            int k
            int c = 0
        m = min(i, j)
        M = max(i, j)
        return self.common_neighbors[m * self.num_edges + M - ((m * (m + 1))//2)]

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @nonecheck(False)
    @staticmethod
    cdef long* common_neighbors_matrix(short[:, :] incidence_matrix) nogil:
        cdef:
            long* common_neighbors
            int c
            int i, j, k
            int m, M
            int num_nodes, num_edges

        num_nodes = incidence_matrix.shape[0]
        num_edges = incidence_matrix.shape[1]

        common_neighbors = <long*> malloc((((num_edges) * (num_edges + 1)) / 2) * sizeof(long))
        for i in prange(num_edges):
            for j in range(i, num_edges):
                c = 0
                m = min(i, j)
                M = max(i, j)
                for k in range(num_nodes):
                    c = c + incidence_matrix[k][i] * incidence_matrix[k][j]
                common_neighbors[m * num_edges + M - ((m * (m + 1))/2)] = c
        return common_neighbors

# =============================================================================

cdef struct Node:
    long index
    int _id, i, j, k
    Node *next

cdef class Mochy:

    cdef CommonNeighbors common_neighbors
    cdef short[:, :] incidence_matrix
    cdef short* incidence_matrix_
    cdef int optimized
    cdef int num_nodes
    cdef int num_edges
    cdef long[30] motif_counts
    cdef int counted

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @nonecheck(False)
    def __cinit__(self, short[:, :] incidence_matrix, int optimized = 1):
        cdef:
            int i, j, k
        self.optimized = optimized
        self.common_neighbors = None
        self.num_nodes = incidence_matrix.shape[0]
        self.num_edges = incidence_matrix.shape[1]
        self.counted = 0
        self.incidence_matrix = incidence_matrix
        self.incidence_matrix_ = <short*> malloc(incidence_matrix.shape[0] * incidence_matrix.shape[1] * sizeof(short))
        for i in range(incidence_matrix.shape[0]):
            for j in range(incidence_matrix.shape[1]):
                self.incidence_matrix_[i * self.num_edges + j] = incidence_matrix[i][j]

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @nonecheck(False)
    cpdef cnp.ndarray[cnp.int32_t, ndim=2] sample(self, int h = -1, int limit = -1):
        cdef:
            int i, j, k
            int a, b, c, d, e, f, g
            double p, random
            int c_ij, c_ik, c_jk, g_ijk
            int deg_i, deg_j, deg_k
            int num_threads, thread_id
            int motif_id, motif_index
            int[128] id_to_index
            Node **head = NULL
            Node **tail = NULL
            Node *node
            Node *temp_node
            double[30] sample_probabilities
            int[:, :] motifs
            int motifsloc = 0
        id_to_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 23, 22, 24, 23, 25, 24, 26, 0, 0, 0, 0, 0, 0, 0, 0, 21, 22, 23, 24, 23, 24, 25, 26, 21, 23, 23, 25, 22, 24, 24, 26, 27, 28, 28, 29, 28, 29, 29, 30, 1, 2, 2, 3, 2, 3, 3, 4, 5, 6, 6, 8, 7, 9, 9, 10, 5, 7, 6, 9, 6, 9, 8, 10, 11, 13, 12, 14, 13, 15, 14, 16, 5, 6, 7, 9, 6, 8, 9, 10, 11, 12, 13, 14, 13, 14, 15, 16, 11, 13, 13, 15, 12, 14, 14, 16, 17, 18, 18, 19, 18, 19, 19, 20]
        num_threads = openmp.omp_get_max_threads()
        head = <Node **> malloc(sizeof(Node*) * num_threads)
        tail = <Node **> malloc(sizeof(Node*) * num_threads)
        if self.counted == 0:
            self.count()

        for i in range(30):
            c = self.motif_counts[i]
            if c == 0:
                sample_probabilities[i] = 0
            elif limit == -1:
                sample_probabilities[i] = 1
            else:
                p = <float>limit / c
                if p > 1:
                    sample_probabilities[i] = 1
                else:
                    sample_probabilities[i] = p

        for i in prange(self.num_edges, nogil=True):
            thread_id = threadid()
            deg_i = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, i, i)
            for j in range(self.num_edges):
                if i != j:
                    c_ij = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, i, j)
                    if c_ij > 0:
                        for k in range(j + 1, self.num_edges):
                            if i != k:
                                c_jk = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, j, k)
                                if i < j or c_jk == 0:
                                    c_ik = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, i, k)
                                    if c_ik > 0:
                                        deg_j = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, j, j)
                                        deg_k = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, k, k)
                                        g_ijk = 0
                                        for e in range(self.num_nodes):
                                            if self.incidence_matrix_[e * self.num_edges + i] and self.incidence_matrix_[e * self.num_edges + j] and self.incidence_matrix_[e * self.num_edges + k]:
                                                g_ijk = g_ijk + 1
                                        motif_id = get_motif_id(deg_i, deg_j, deg_k, c_ij, c_ik, c_jk, g_ijk)
                                        motif_index = id_to_index[motif_id] - 1
                                        if h < 1 or motif_index == h:
                                            random = <double>rand() / RAND_MAX
                                            if random < sample_probabilities[motif_index]:
                                                node = <Node *> malloc(sizeof(Node))
                                                node._id = motif_index
                                                node.i = i
                                                node.j = j
                                                node.k = k
                                                if head[thread_id] is NULL:
                                                    node.index = 0
                                                    tail[thread_id] = node
                                                else:
                                                    node.index = head[thread_id].index + 1
                                                node.next = head[thread_id]
                                                head[thread_id] = node


        node = NULL

        for i in range(num_threads - 1):
            if head[i] is NULL:
                continue
            j = i + 1
            while tail[j] is NULL:
                if j == num_threads - 1:
                    break
                j += 1
            if tail[j] is NULL:
                continue
            tail[j].next = head[i]
            head[j].index += head[i].index + 1
            node = head[j]
        
        if node is NULL:
            motifs = np.empty((0, 4), dtype=np.int32)
        else:
            motifs = np.empty((node.index + 1, 4), dtype=np.int32)

        while node is not NULL:
            motifs[motifsloc][0] = node._id
            motifs[motifsloc][1] = node.i
            motifs[motifsloc][2] = node.j
            motifs[motifsloc][3] = node.k
            motifsloc += 1
            temp_node = node
            node = node.next
            free(temp_node)

        return np.array(motifs)

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @nonecheck(False)
    cpdef cnp.ndarray[cnp.int64_t, ndim=1] count(self):
        cdef:
            int i, j, k
            int a, b, c, d, e, f, g
            int c_ij, c_ik, c_jk, g_ijk
            int deg_i, deg_j, deg_k
            int motif_id, motif_index
            long* local_motif_counts
            int[128] id_to_index
        id_to_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 23, 22, 24, 23, 25, 24, 26, 0, 0, 0, 0, 0, 0, 0, 0, 21, 22, 23, 24, 23, 24, 25, 26, 21, 23, 23, 25, 22, 24, 24, 26, 27, 28, 28, 29, 28, 29, 29, 30, 1, 2, 2, 3, 2, 3, 3, 4, 5, 6, 6, 8, 7, 9, 9, 10, 5, 7, 6, 9, 6, 9, 8, 10, 11, 13, 12, 14, 13, 15, 14, 16, 5, 6, 7, 9, 6, 8, 9, 10, 11, 12, 13, 14, 13, 14, 15, 16, 11, 13, 13, 15, 12, 14, 14, 16, 17, 18, 18, 19, 18, 19, 19, 20]
        if self.counted == 0:
            self.common_neighbors = CommonNeighbors(self.incidence_matrix, self.num_edges, self.optimized)
            for i in prange(self.num_edges, nogil=True):
                local_motif_counts = <long*> malloc(30 * sizeof(long))
                for j in range(30):
                    local_motif_counts[j] = 0
                deg_i = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, i, i)
                for j in range(self.num_edges):
                    if i != j:
                        c_ij = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, i, j)
                        if c_ij > 0:
                            for k in range(j + 1, self.num_edges):
                                if i != k:
                                    c_jk = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, j, k)
                                    if i < j or c_jk == 0:
                                        c_ik = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, i, k)
                                        if c_ik > 0:
                                            deg_j = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, j, j)
                                            deg_k = get_common_neighbor(self.common_neighbors.common_neighbors, self.num_edges, k, k)
                                            g_ijk = 0
                                            for e in range(self.num_nodes):
                                                if self.incidence_matrix_[e * self.num_edges + i] and self.incidence_matrix_[e * self.num_edges + j] and self.incidence_matrix_[e * self.num_edges + k]:
                                                    g_ijk = g_ijk + 1
                                            motif_id = get_motif_id(deg_i, deg_j, deg_k, c_ij, c_ik, c_jk, g_ijk)
                                            motif_index = id_to_index[motif_id] - 1
                                            local_motif_counts[motif_index] = local_motif_counts[motif_index] + 1
                for i in range(30):
                    self.motif_counts[i] += local_motif_counts[i]
                free(local_motif_counts)
            self.counted = 1
        return np.array(self.motif_counts)
