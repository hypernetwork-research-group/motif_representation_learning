import numpy as np
from sklearn.metrics import roc_curve

def sensivity_specifity_cutoff(y_true, y_score):
    '''Find data-driven cut-off for classification
    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    References
    ----------
    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

from sklearn.model_selection import train_test_split

def incidence_matrix_train_test_split(incidence_matrix: np.ndarray, train_size: float):
    '''Split incidence matrix into train and test sets
    
    Parameters
    ----------
    
    incidence_matrix : array, shape = [n_samples, n_features]
        Incidence matrix.
        
    train_size : float
        Proportion of the dataset to include in the train split.
        
    Returns
    -------
    
    train_incidence_matrix : array, shape = [n_samples, n_features]
        Train incidence matrix.
        
    test_incidence_matrix : array, shape = [n_samples, n_features]
        Test incidence matrix.
    '''
    T_incidence_matrix, t_incidence_matrix = (im.T for im in train_test_split(incidence_matrix.T, train_size=0.8))
    t_incidence_matrix = np.hstack((T_incidence_matrix, t_incidence_matrix)) # The test incidence matrix is the union of the known edges and the unknown edges
    return T_incidence_matrix, t_incidence_matrix

def incidence_matrix_fold_extract(incidence_matrix: np.ndarray, train_index, test_index):
    T_incidence_matrix, t_incidence_matrix = incidence_matrix.T[train_index].T, incidence_matrix.T[test_index].T
    t_incidence_matrix = np.hstack((T_incidence_matrix, t_incidence_matrix))
    return T_incidence_matrix, t_incidence_matrix

from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def evaluate_estimator(estimator: BaseEstimator, incidence_matrix, motifs, y_true, threshold: float = None):
    y_pred_m = estimator.predict(incidence_matrix, motifs)
    if threshold is None:
        threshold = sensivity_specifity_cutoff(y_true, y_pred_m)
        if threshold == np.inf:
            threshold = 0.5
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_m),
        'accuracy': accuracy_score(y_true, y_pred_m > threshold),
        'f1': f1_score(y_true, y_pred_m > threshold),
        'threshold': threshold,
    }
    return metrics

from sklearn.base import BaseEstimator

class CustomEstimator(BaseEstimator):

    pass

from torch.utils.data import Dataset

class MotifIteratorDataset(Dataset):
    def __init__(self, incidence_matrix, attributes):
        """
        incidence_matrix: Tensor che rappresenta la matrice di incidenza del grafo.
        attributes: Tensor contenente attributi o etichette che necessitano di batching.
        """
        self.incidence_matrix = incidence_matrix
        self.attributes = attributes

    def __len__(self):
        # La lunghezza del dataset è determinata dal numero di elementi in `attributes`
        return len(self.attributes)

    def __getitem__(self, idx):
        # Restituisce la matrice di incidenza completa e l'elemento degli attributi all'indice idx
        return self.incidence_matrix, self.attributes[idx]
