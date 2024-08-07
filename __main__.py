import numpy as np
from pymochy import Mochy
from motif import motif_negative_sampling
from datasets import EmailEnronFull, ContactHighSchool, ContactPrimarySchool, Cora
from models.adamic_adar import AdamicAdar
from models.jaccard_coefficient import JaccardCoefficient
from models.common_neighbors import CommonNeighors
from models.hypergraph_motif_conv import HypergraphMotifConv
from models.hgmrl_villain import HypergraphMotifConvVilLain
from models.node2vec_hypergcn import Node2VecHyperGCN
from models.node2vec import Node2Vec
from models.hpra import HPRA
from models.villain import VilLainSLP
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from utils import incidence_matrix_train_test_split, incidence_matrix_fold_extract
from utils import evaluate_estimator
from rich.logging import RichHandler
import pickle
import os

import logging

import os
os.environ['OMP_NUM_THREADS'] = '128'

from clearml import Task

def main(dataset: Dataset, models: dict[str, BaseEstimator], k: int, limit: int, alpha: float, beta: int):
    print(dataset)
    print(f"k = {k})")

    incidence_matrix = dataset.incidence_matrix(lambda e: len(e) > 1)
    print(incidence_matrix.shape)

    experiments_metrics = { model_name: [] for model_name in models.keys()}

    # current_logger = Logger.current_logger()

    for i in range(3): # Experiments iteration
        logging.info(f"Experiment {i}")

        model_metrics = { model_name: [] for model_name in models.keys()}
        model_params = { model_name: [] for model_name in models.keys() }

        while True:
            T_incidence_matrix, t_incidence_matrix = incidence_matrix_train_test_split(incidence_matrix, 0.8)


            t_mochy = Mochy(t_incidence_matrix)
            T_mochy = Mochy(T_incidence_matrix)
            
            t_motif_count = t_mochy.count()
            T_motif_count = T_mochy.count()

            t_motifs = t_mochy.sample(h=k, limit=limit)[:, 1:]
            T_motifs = T_mochy.sample(h=k, limit=limit)[:, 1:]
            t_motifs = t_motifs[~np.isin(t_motifs, T_motifs).all(axis=1)]

            logging.info(f"Training motifs: {T_motifs.shape[0]}/{T_motif_count[k]}")
            logging.info(f"Test motifs: {t_motifs.shape[0]}/{t_motif_count[k]}")

            if T_motifs.shape[0] > 0 and t_motifs.shape[0] > 0:
                break
            print(f"[Retrying] To few samples for Training or Testing")

        kf = KFold(n_splits=5)
        for f, (train_index, test_index) in enumerate(kf.split(T_incidence_matrix.T)): # k-fold cross validation
            logging.debug(f"Fold {f}")
            V_incidence_matrix, v_incidence_matrix = incidence_matrix_fold_extract(T_incidence_matrix, train_index, test_index)

            # Mochy is a class that allows to sample motifs from an incidence matrix
            V_mochy = Mochy(V_incidence_matrix)
            v_mochy = Mochy(v_incidence_matrix)

            F_motif_count = V_mochy.count()
            f_motif_count = v_mochy.count()

            F_motifs = V_mochy.sample(h=k, limit=limit)[:, 1:]              # Training motifs
            f_motifs = v_mochy.sample(h=k, limit=limit)[:, 1:]              # All the motifs in the fold
            v_motifs = f_motifs[~np.isin(f_motifs, F_motifs).all(axis=1)]   # Validation motifs are the motifs that are not in the training motifs

            logging.debug(f"Training motifs: {F_motifs.shape[0]}/{F_motif_count[k]}")
            logging.debug(f"Validation motifs: {v_motifs.shape[0]}/{f_motif_count[k]}")

            # Negative sampling
            v_incidence_matrix, v_motifs, y_validation_e, y_validation_m = motif_negative_sampling(v_incidence_matrix, v_motifs, alpha, beta, mode=os.environ['NGTV_MODE'])

            for model_name, Model in models.items():
                model = Model()
                try:
                    model.fit(V_incidence_matrix, F_motifs, v_incidence_matrix, y_validation_e, v_motifs, y_validation_m)

                    metrics = evaluate_estimator(model, v_incidence_matrix, v_motifs, y_validation_m)
                    logging.debug(f"{model_name} {metrics}")
                    model_metrics[model_name].append(metrics)

                    model_params[model_name].append(model.get_params()) # Save the model parameters
                except Exception as e:
                    logging.error(f"{model_name} {e}")

        t_incidence_matrix, t_motifs, y_test_e, y_test_m = motif_negative_sampling(t_incidence_matrix, t_motifs, alpha, beta, mode=os.environ['NGTV_MODE'])

        os.environ['N_EXPERIMENT'] = f'{i}'

        for model_name, Model in models.items():
            model = Model()
            model.fit(T_incidence_matrix, T_motifs, t_incidence_matrix, y_test_e, t_motifs, y_test_m)

            threshold = np.mean([metric['threshold'] for metric in model_metrics[model_name]])
            metrics = evaluate_estimator(model, t_incidence_matrix, t_motifs, y_test_m, threshold)
            logging.info(f"{model_name} {metrics}")

            experiments_metrics[model_name].append(metrics)


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

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument('--dataset', type=str, choices=['email_enron', 'contact_high_school', 'contact_primary_school', 'congress_bills', 'cora', 'pubmed'])
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--mode', type=str, choices=['rank', 'random', 'prob'], default='rank')

    args = parser.parse_args()

    if args.dataset == 'email_enron':
        dataset = EmailEnronFull()
    elif args.dataset == 'contact_high_school':
        dataset = ContactHighSchool()
    elif args.dataset == 'contact_primary_school':
        dataset = ContactPrimarySchool()
    elif args.dataset == 'congress_bills':
        dataset = CongressBillsFull()
    elif args.dataset == 'cora':
        dataset = Cora()
    elif args.dataset == 'pubmed':
        dataset = PubMed()
    elif args.dataset == 'citeseer':
        dataset = Citeseer()

    incidence_matrix = dataset.incidence_matrix(lambda e: len(e) > 1)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler()]
    )

    os.environ['NGTV_MODE'] = args.mode

    models = dict()
    models['Hypergraph Motif Conv VilLain'] = HypergraphMotifConvVilLain
    models['Jaccard Coefficient'] = JaccardCoefficient
    models['Hypergraph Motif Conv'] = HypergraphMotifConv
    models['VilLain'] = VilLainSLP
    models['Node2Vec'] = Node2Vec
    models['Node2Vec HyperGCN'] = Node2VecHyperGCN
    models['HPRA'] = HPRA
    models['Adamic Adar'] = AdamicAdar
    models['Common Neighors'] = CommonNeighors

    k = args.k
    limit = args.limit

    main(dataset, models, k, limit, 0.5, 1)
