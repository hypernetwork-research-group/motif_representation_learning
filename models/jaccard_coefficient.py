import numpy as np
from itertools import combinations
from utils import CustomEstimator

class JaccardCoefficient(CustomEstimator):

    def fit(self, X: np.ndarray, *args, **kwargs):
        X_ = X
        A = X_ @ X_.T # Compute the clique expansion of the hypergraph
        D = np.diag(np.diag(A))
        A = np.where(A - D, 1, 0)
        # self.G = nx.from_pandas_adjacency(pd.DataFrame(A))
        intersections = np.dot(A, A)
        degrees = A.sum(axis=1)
        union = degrees[:, None] + degrees - intersections
        with np.errstate(divide='ignore', invalid='ignore'):
            jaccard = intersections / union
            # Imposta i valori della diagonale principale a 0 per evitare divisione per zero
            np.fill_diagonal(jaccard, 0)
        jaccard = np.where(np.isnan(jaccard), 0, jaccard)

        # Normalize the jaccard scores
        self.c = np.ceil(np.linalg.norm(jaccard, np.inf))

        self.jaccard = jaccard

    def predict(self, X: np.ndarray, motifs: np.ndarray):
        y_pred_e = np.zeros(X.shape[1])
        # edge_list = [np.where(hyperedge)[0] for hyperedge in X.T]
        for i, e in enumerate(X.T):
            if e.sum() <= 1:
                y_pred_e[i] = 0
                continue
            edge_indexes = np.array(list(combinations(list(e.nonzero()[0]), 2))).T
            jc = self.jaccard[edge_indexes[0, :], edge_indexes[1, :]]
            jc = np.where(np.isnan(jc), 0, jc)
            y_pred_e[i] = jc.mean()
        y_pred_e = y_pred_e.reshape(-1, 1)
        y_pred_m = y_pred_e[motifs].prod(axis=1)
        y_pred_m = y_pred_m / self.c
        return y_pred_m

    def _more_tags(self):
        return {'requires_y': False}

