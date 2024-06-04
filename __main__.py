import numpy as np
from pymochy import Mochy
from motif import motif_negative_sampling
from datasets import EmailEnronFull, EmailEuFull, ContactHighSchool, ContactPrimarySchool, NDCClassesFull, TagsAskUbuntu, TagsMathSx
from models.jaccard_coefficient import JaccardCoefficient
from models.adamic_adar import AdamicAdar
from models.common_neighbors import CommonNeighors
from models.hypergraph_motif_conv import HypergraphMotifConv
from datasets import Dataset
from time import time
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from utils import incidence_matrix_train_test_split, incidence_matrix_fold_extract
from utils import evaluate_estimator

import seaborn as sns
import matplotlib.pyplot as plt

import os
os.environ['OMP_NUM_THREADS'] = '128'

def main(dataset: Dataset, models: dict[str, BaseEstimator], k: int, limit: int, alpha: float, beta: int):
    print(dataset)

    incidence_matrix = dataset.incidence_matrix(lambda e: len(e) > 1)

    experiments_metrics = { model_name: [] for model_name in models.keys()}

    for i in range(3): # Experiments iteration
        begin = time()
        print(f"Experiment {i}")
        T_incidence_matrix, t_incidence_matrix = incidence_matrix_train_test_split(incidence_matrix, 0.8)

        model_metrics = { model_name: [] for model_name in models.keys()}
        model_params = { model_name: [] for model_name in models.keys() }

        t_mochy = Mochy(t_incidence_matrix)
        T_mochy = Mochy(T_incidence_matrix)

        t_motif_count = t_mochy.count()
        T_motif_count = T_mochy.count()

        t_motifs = t_mochy.sample(h=k, limit=limit)[:, 1:]
        T_motifs = T_mochy.sample(h=k, limit=limit)[:, 1:]
        t_motifs = t_motifs[~np.isin(t_motifs, T_motifs).all(axis=1)]

        kf = KFold(n_splits=5)
        for f, (train_index, test_index) in enumerate(kf.split(T_incidence_matrix.T)): # k-fold cross validation
            print(f"Fold {f}")
            V_incidence_matrix, v_incidence_matrix = incidence_matrix_fold_extract(T_incidence_matrix, train_index, test_index)

            # Mochy is a class that allows to sample motifs from an incidence matrix
            V_mochy = Mochy(V_incidence_matrix)
            v_mochy = Mochy(v_incidence_matrix)

            F_motif_count = V_mochy.count()
            f_motif_count = v_mochy.count()

            F_motifs = V_mochy.sample(h=k, limit=limit)[:, 1:]              # Training motifs
            f_motifs = v_mochy.sample(h=k, limit=limit)[:, 1:]              # All the motifs in the fold
            v_motifs = f_motifs[~np.isin(f_motifs, F_motifs).all(axis=1)]   # Validation motifs are the motifs that are not in the training motifs

            # Negative sampling
            v_incidence_matrix, v_motifs, y_validation_e, y_validation_m = motif_negative_sampling(v_incidence_matrix, v_motifs, alpha, beta)
            print(f"Negative sampling done in {time() - begin} seconds")

            for model_name, Model in models.items():
                model = Model()
                model.fit(V_incidence_matrix, F_motifs)

                metrics = evaluate_estimator(model, v_incidence_matrix, v_motifs, y_validation_m)
                print(model_name, metrics)
                model_metrics[model_name].append(metrics)

                model_params[model_name].append(model.get_params()) # Save the model parameters

        t_incidence_matrix, t_motifs, y_test_e, y_test_m = motif_negative_sampling(t_incidence_matrix, t_motifs, alpha, beta)

        # Test the models
        for model_name, Model in models.items():
            model = Model()
            model.fit(T_incidence_matrix, T_motifs)

            threshold = np.mean([metrics['threshold'] for metrics in model_metrics[model_name]])
            metrics = evaluate_estimator(model, t_incidence_matrix, t_motifs, y_test_m, threshold)

            experiments_metrics[model_name].append(metrics)
            print(model_name, metrics)
        
        end = time()
        print("Experiment time:", end - begin)

    print("=====")

    for model_name, metrics in experiments_metrics.items():
        print(model_name)
        roc_auc = [metric['roc_auc'] for metric in metrics]
        print("ROC AUC:", f"{np.min(roc_auc):.2}", "<", f"{np.mean(roc_auc):.2}", "+-", f"{np.std(roc_auc):.2}", "<", f"{np.max(roc_auc):.2}")
        accuracy = [metric['accuracy'] for metric in metrics]
        print("Accuracy:", f"{np.min(accuracy):.2}", "<", f"{np.mean(accuracy):.2}", "+-", f"{np.std(accuracy):.2}", "<", f"{np.max(accuracy):.2}")
        f1 = [metric['f1'] for metric in metrics]
        print("F1:", f"{np.min(f1):.2}", "<", f"{np.mean(f1):.2}", "+-", f"{np.std(f1):.2}", "<", f"{np.max(f1):.2}")
        threshold = [metric['threshold'] for metric in metrics]
        print("Threshold:", f"{np.min(threshold):.2}", "<", f"{np.mean(threshold):.2}", "+-", f"{np.std(threshold):.2}", "<", f"{np.max(threshold):.2}")
        print("=====")

if __name__ == '__main__':
    dataset = EmailEnronFull()
    incidence_matrix = dataset.incidence_matrix(lambda e: len(e) > 1)

    models = dict()
    models['Hypergraph Motif Conv'] = HypergraphMotifConv
    models['Jaccard Coefficient'] = JaccardCoefficient
    models['Adamic Adar'] = AdamicAdar
    models['Common Neighors'] = CommonNeighors

    main(dataset, models, 29, 25000, 0.5, 1)
