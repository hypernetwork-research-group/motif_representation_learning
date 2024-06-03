import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import combinations
from utils import CustomEstimator

class CommonNeighors(CustomEstimator):

    def fit(self, X: np.ndarray, motifs: np.ndarray):
        X_ = X
        D = np.diag(X_.sum(axis=1))
        A = (X_ @ X_.T) - D
        A = np.where(A, 1, 0)
        # A is the adjacency matrix of the hypergraph clique expansion
        common_neighbors = A @ A
        self.common_neighbors = common_neighbors - np.diag(np.diag(common_neighbors)) # Remove the diagonal
        # Normalize the common neighbors scores
        self.common_neighbors = (self.common_neighbors - self.common_neighbors.min()) / (self.common_neighbors.max() - self.common_neighbors.min())
        return self

    def predict(self, X: np.ndarray, motifs: np.ndarray):
        y_pred_e = np.zeros(X.shape[1])
        for i, e in enumerate(X.T):
            edge_indexes = np.array(list(combinations(list(e.nonzero()[0]), 2))).T
            cn = self.common_neighbors[edge_indexes[0, :], edge_indexes[1, :]]
            cn = np.where(np.isnan(cn), 0, cn)
            y_pred_e[i] = cn.mean()
        y_pred_e = y_pred_e.reshape(-1, 1)
        y_pred_m = y_pred_e[motifs].prod(axis=1)
        return y_pred_m

    def _more_tags(self):
        return {'requires_y': False}
