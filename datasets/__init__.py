import numpy as np
from os import path
from abc import ABC

DATASET_DIR = '__datasets__'

# Implementare questi dataset come dataset torch_geometric

class Dataset(ABC):

    DATASET_NAME = None
    HAS_NODE_LABELS = False

    def __init__(self):
        module_dir = path.dirname(path.abspath(__file__))
        dataset_dir = path.join(module_dir, DATASET_DIR)
        if self.HAS_NODE_LABELS:
            self.node_labels = {label[0]:label[1] for label in map(str.split, open(f'{dataset_dir}/{self.DATASET_NAME}/{self.DATASET_NAME}-node-labels.txt', 'r').readlines())}
        self.times = list(map(int, open(f'{dataset_dir}/{self.DATASET_NAME}/{self.DATASET_NAME}-times.txt', 'r').readlines()))
        self.nverts = list(map(int, open(f'{dataset_dir}/{self.DATASET_NAME}/{self.DATASET_NAME}-nverts.txt', 'r').readlines()))
        self.simplices = list(map(int, open(f'{dataset_dir}/{self.DATASET_NAME}/{self.DATASET_NAME}-simplices.txt', 'r').readlines()))
    
    @property
    def edge_list(self):
        if getattr(self, '_edge_list', None) is None:
            j = 0
            edge_list = []
            for nvert in self.nverts:
                edge = []
                for _ in range(nvert):
                    edge.append(self.simplices[j])
                    j += 1
                edge_list.append(edge)
            self._edge_list = edge_list
        return self._edge_list

    @property
    def unique_edges(self):
        if getattr(self, '_unique_edges', None) is None:
            self._unique_edges = list(map(list, set([tuple(sorted(edge)) for edge in self.edge_list])))
        return self._unique_edges

    @property
    def num_vertices(self):
        return max(map(max, self.edge_list)) + 1

    def __str__(self):
        return f'{self.DATASET_NAME} dataset: {len(self.times)} time steps, {self.num_vertices} vertices, {len(self.unique_edges)} unique edges'

    def incidence_matrix(self, filter=None):
        num_vertices = self.num_vertices
        unique_edges = [e for e in self.unique_edges if filter is None or filter(e)]
        num_edges = len(unique_edges)
        incidence_matrix = np.zeros((num_vertices, num_edges))
        for i, edge in enumerate(unique_edges):
            incidence_matrix[edge, i] = 1
        return incidence_matrix.astype(np.int16)

    @property
    def edge_index(self):
        return self.incidence_matrix().nonzero()

class EmailEnronFull(Dataset):

    DATASET_NAME = 'email-Enron-full'
    HAS_NODE_LABELS = True

class ContactHighSchool(Dataset):

    DATASET_NAME = 'contact-high-school'
    HAS_NODE_LABELS = False

class ContactPrimarySchool(Dataset):

    DATASET_NAME = 'contact-primary-school'
    HAS_NODE_LABELS = False

class Cora(Dataset):

    DATASET_NAME = 'cora'
    HAS_NODE_LABELS = False
