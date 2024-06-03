import numpy as np
from utils import CustomEstimator

class AdamicAdar(CustomEstimator):

    def fit(self, X: np.ndarray, motifs: np.ndarray):
        X_ = X
        D = np.diag(X_.sum(axis=1))
        A = (X_ @ X_.T) - D
        A = np.where(A, 1, 0)
        # A is the adjacency matrix of the hypergraph clique expansion
        degrees = A.sum(axis=1)
        with np.errstate(divide='ignore'):
            inv_log_degrees = 1 / np.log(degrees)
        inv_log_degrees[np.isinf(inv_log_degrees)] = 0

        adamic_adar = np.zeros_like(A)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i != j:
                    if A[i, j]:
                        common_neighbors = np.where(A[i] & A[j])[0]
                        adamic_adar[i, j] = np.sum(inv_log_degrees[common_neighbors])
        
        # Normalize the adamic adar scores
        adamic_adar = (adamic_adar - adamic_adar.min()) / (adamic_adar.max() - adamic_adar.min())

        self.adamic_adar = adamic_adar

    def predict(self, X: np.ndarray, motifs: np.ndarray):
        y_pred_e = np.zeros(X.shape[1])
        for i, e in enumerate(X.T):
            edge_indexes = np.where(e)[0]
            aa = self.adamic_adar[edge_indexes[0], edge_indexes[1]]
            y_pred_e[i] = aa.mean()
        y_pred_e = y_pred_e.reshape(-1, 1)
        y_pred_m = y_pred_e[motifs].prod(axis=1)
        return y_pred_m

    def _more_tags(self):
        return {'requires_y': False}
