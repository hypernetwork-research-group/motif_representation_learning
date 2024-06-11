import numpy as np
import scipy.sparse as sp
from utils import CustomEstimator


def get_adj_matrix(H):

    # computing inverse hyedge degree Matrix
    d_e = np.subtract(np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=0))), 1)
    D_e_inv = sp.spdiags(np.reciprocal(d_e), [0], H.shape[1], H.shape[1], format="csr")

    # computing node degree preserving reduction's adjacency matrix
    A_ndp = H.dot(D_e_inv.dot(H.transpose()))
    A_ndp = A_ndp - sp.spdiags(A_ndp.diagonal(), [0], A_ndp.shape[0], A_ndp.shape[1], format="csr")

    return A_ndp

class HPRA(CustomEstimator):

    def fit(self, X: np.ndarray, *args, **kwargs):
        csr = sp.csr_matrix(X)
        sp.csr_matrix.sum(csr, axis=1)
        d_v = np.squeeze(np.asarray(sp.csr_matrix.sum(csr, axis=1)))
        with np.errstate(divide='ignore'):
            d_v_inv = 1. / d_v
        d_v_inv[d_v_inv == np.inf] = 0
        D_v_inv = sp.diags(d_v_inv)

        A_ndp = get_adj_matrix(csr)

        hra_scores = A_ndp + (A_ndp.dot(D_v_inv)).dot(A_ndp).toarray()

        self.c = np.ceil(np.linalg.norm(hra_scores, np.inf))

        self.hra_scores = hra_scores


    def predict(self, X: np.ndarray, motifs: np.ndarray):
        y_pred_e = np.zeros(X.shape[1])
        for i, e in enumerate(X.T):
            edge_indexes = e.nonzero()[0]
            jc = self.hra_scores[edge_indexes, :][:, edge_indexes]
            jc = np.where(np.isnan(jc), 0, jc)
            y_pred_e[i] = jc.mean()
        y_pred_e = y_pred_e.reshape(-1, 1)
        y_pred_m = y_pred_e[motifs].prod(axis=1)
        y_pred_m = y_pred_m / self.c
        return y_pred_m

