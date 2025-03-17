import numpy as np

import torch
from torch import nn
import torch_geometric.nn.aggr
from torch_geometric.nn import Node2Vec, HypergraphConv, GCNConv

class Node2VecHypergraphConv(nn.Module): # Da testare se da le stesse performance

    def __init__(self, channels, edge_index, num_nodes) -> None:
        super().__init__()
        self.node2vec = Node2Vec(edge_index, embedding_dim=channels, walk_length=80, context_size=80, walks_per_node=10, num_negative_samples=1, p=0.5, q=2, sparse=False, num_nodes=num_nodes)
        self.hypergraph_node_conv = HypergraphConv(channels, channels, attention_mode="edge")
        self.linear = nn.Linear(channels, channels)

    def forward(self, edge_index):
        y = self.node2vec()
        y = self.hypergraph_node_conv(y, edge_index)
        y = nn.functional.leaky_relu(y)
        y = y.T @ y
        y = self.linear(y)
        y = nn.functional.leaky_relu(y) # TODO: Remove this
        return y

class HypergraphMotifConvE(nn.Module):

    def __init__(self, channels_1, channels_2, channels_3, channels_out, edge_index, num_nodes) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.5)

        self.node2vec_hypergraph_conv = Node2VecHypergraphConv(channels_1, edge_index, num_nodes)

        self.aggr_1 = torch_geometric.nn.MeanAggregation()
        self.hypergraph_edge_conv_1 = GCNConv(channels_1, channels_2)
        self.linear_2 = nn.Linear(channels_2, channels_2)
        self.edge_linear_out = nn.Linear(channels_2, channels_out)

        self.aggr_2 = torch_geometric.nn.aggr.MinAggregation()
        self.linear_3 = nn.Linear(channels_2, channels_3)

        self.linear_out = nn.Linear(channels_3, channels_out)
    
    def node_embeddings(self, edge_index):
        y = self.node2vec_hypergraph_conv(edge_index)
        return y

    def edge_embeddings(self, X, edge_index, edge_edge_index, sigmoid=False):
        y = X
        y = self.aggr_1(y[edge_index[0]], edge_index[1])
        y = self.dropout(y)
        y = self.hypergraph_edge_conv_1(y, edge_edge_index)
        y = self.linear_2(y)
        y = nn.functional.leaky_relu(y)
        y_out = self.edge_linear_out(y)
        if sigmoid:
            y_out = nn.functional.sigmoid(y_out)
        return y, y_out

    def motif_embeddings(self, X, motif_edge_index, sigmoid=False):
        y = X
        y = self.aggr_2(y[motif_edge_index[0]], motif_edge_index[1])
        y = self.dropout(y)
        y = self.linear_3(y)
        y_out = self.linear_out(y)
        if sigmoid:
            y_out = nn.functional.sigmoid(y_out)
        return y, y_out

    def forward(self, edge_index, edge_edge_index, motif_edge_index, sigmoid=False):
        y = self.node_embeddings(edge_index)
        y, _ = self.edge_embeddings(y, edge_index, edge_edge_index)
        y, y_out = self.motif_embeddings(y, motif_edge_index, sigmoid)
        return y_out

# ------------------------------------------------------

from utils import CustomEstimator
from motif import motif_negative_sampling, edge_motif_interactions, node_node_interactions
import logging
from clearml import Logger
from utils import MotifIteratorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import os

class HypergraphMotifConv(CustomEstimator):

    def __init__(self, batch_size=1000) -> None:
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, motifs: np.ndarray, X_validation, y_validation_e, motifs_validation, y_validation_m):
        num_nodes = X.shape[0]
        nni = torch.tensor(node_node_interactions(X))
        eei = torch.tensor(node_node_interactions(X.T, 3))

        self.eei = eei

        self.model = HypergraphMotifConvE(X.shape[0], 64, 64, 1, nni, num_nodes)
        self.model.train()
        current_logger = Logger.current_logger()

        epochs = 200
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            self.model.train()

            dataset = MotifIteratorDataset(torch.tensor(X), torch.tensor(motifs))
            training_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            loss_e_sum = 0
            loss_m_sum = 0
            for batch in training_loader:
                X_batch = batch[0][0]
                motifs_batch = batch[1]

                X_, motifs_, y_e, y_m = motif_negative_sampling(X_batch.cpu().detach().numpy(), motifs_batch.cpu().detach().numpy(), 0.5, 1, mode=os.environ['NGTV_MODE'], heur=os.environ['NGTV_HEUR'])
                y_m = torch.tensor(y_m)
                nei = torch.tensor(np.array(X_.nonzero()))
                emi = torch.tensor(edge_motif_interactions(X_, motifs_))

                optimizer.zero_grad()
                y = self.model.node_embeddings(nei)
                y, y_pred_e = self.model.edge_embeddings(y, nei, eei)
                _, y_pred_m = self.model.motif_embeddings(y, emi)
                loss_e = criterion(y_pred_e, torch.tensor(y_e))
                loss_m = criterion(y_pred_m, y_m)
                loss_m.backward()
                loss_m_sum += loss_m.item()
                loss_e_sum += loss_e.item()
                optimizer.step()
                logging.debug(f"Epoch {epoch} - Loss_m: {loss_m.item()}")
            # current_logger.report_scalar(title="Loss M", series="HGMRL-m Train", iteration=epoch, value=loss_m_sum / len(training_loader))
            # current_logger.report_scalar(title="Loss E", series="HGMRL-e Train", iteration=epoch, value=loss_e_sum / len(training_loader))

            if epoch % 2 == 0:
                with torch.no_grad():
                    self.model.eval()
                    nei = torch.tensor(np.array(X_validation.nonzero()))
                    emi = torch.tensor(edge_motif_interactions(X_validation, motifs_validation))
                    y = self.model.node_embeddings(nei)
                    y, y_pred_e = self.model.edge_embeddings(y, nei, eei)
                    _, y_pred_m = self.model.motif_embeddings(y, emi)
                    loss_m = criterion(y_pred_m, torch.tensor(y_validation_m))
                    loss_e = criterion(y_pred_e, torch.tensor(y_validation_e))
                    # current_logger.report_scalar(title="Loss M", series="HGMRL-m Validation", iteration=epoch, value=loss_m.item())
                    # current_logger.report_scalar(title="Loss E", series="HGMRL-e Validation", iteration=epoch, value=loss_e.item())
                    roc_auc_m = roc_auc_score(y_validation_m, y_pred_m.cpu().detach().numpy())
                    roc_auc_e = roc_auc_score(y_validation_e, y_pred_e.cpu().detach().numpy())
                    # current_logger.report_scalar(title="ROC AUC", series="HGMRL-m", iteration=epoch, value=roc_auc_m)
                    # current_logger.report_scalar(title="ROC AUC E", series="HGMRL-e", iteration=epoch, value=roc_auc_e)
                    logging.debug(f"Validation Loss_m: {loss_m.item()}")

    def predict(self, X: np.ndarray, motifs: np.ndarray):
        self.model.eval()
        nei = torch.tensor(np.array(X.nonzero()))
        eei = self.eei
        emi = torch.tensor(edge_motif_interactions(X, motifs))

        y = self.model.node_embeddings(nei)
        # pickle.dump(y, open(f"{os.environ['N_EXPERIMENT']}_{os.environ['NGTV_MODE']}_hgmrl_node_embeddings.pkl", "wb"))
        y, y_pred_e = self.model.edge_embeddings(y, nei, eei)
        # pickle.dump(y, open(f"{os.environ['N_EXPERIMENT']}_{os.environ['NGTV_MODE']}_hgmrl_edge_embeddings.pkl", "wb"))
        # pickle.dump(y_pred_e, open(f"{os.environ['N_EXPERIMENT']}_{os.environ['NGTV_MODE']}_hgmrl_edge_predictions.pkl", "wb"))
        y, y_pred_m = self.model.motif_embeddings(y, emi, sigmoid=True)
        # pickle.dump(y, open(f"{os.environ['N_EXPERIMENT']}_{os.environ['NGTV_MODE']}_hgmrl_motif_embeddings.pkl", "wb"))
        # pickle.dump(y_pred_m, open(f"{os.environ['N_EXPERIMENT']}_{os.environ['NGTV_MODE']}_hgmrl_motif_predictions.pkl", "wb"))
        y_pred_m = y_pred_m.cpu().detach().numpy()
        return y_pred_m
