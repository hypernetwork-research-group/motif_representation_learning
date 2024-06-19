import numpy as np
import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import Node2Vec as N2V
from utils import CustomEstimator

class _Node2Vec(nn.Module):

    def __init__(self, in_features:int, out_features: int, edge_index: torch.Tensor, num_nodes: int):
        super(_Node2Vec, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.node2vec = N2V(edge_index, embedding_dim=in_features, walk_length=80, context_size=80, walks_per_node=10, num_negative_samples=1, p=0.5, q=2, sparse=False, num_nodes=num_nodes)
        self.linear = nn.Linear(in_features, out_features)
        self.aggr_1 = torch_geometric.nn.aggr.MeanAggregation()
        self.aggr_2 = torch_geometric.nn.aggr.MinAggregation()
    
    def forward(self, edge_index: torch.Tensor, emi: torch.Tensor, sigmoid: bool = False) -> torch.Tensor:
        y = self.node2vec()
        y = self.dropout(y)
        y = nn.functional.leaky_relu(y)
        y = y.T @ y
        y = self.linear(y)
        y = nn.functional.leaky_relu(y)
        y_e = self.aggr_1(y[edge_index[0]], edge_index[1])
        y_m = self.aggr_2(y_e[emi[0]], emi[1])
        if sigmoid:
            y_e = nn.functional.sigmoid(y_e)
            y_m = nn.functional.sigmoid(y_m)
        return y_e, y_m

from utils import MotifIteratorDataset
from torch.utils.data import DataLoader
from motif import motif_negative_sampling, edge_motif_interactions, node_node_interactions
import logging
from clearml import Logger
from sklearn.metrics import roc_auc_score

class Node2Vec(CustomEstimator):

    def __init__(self, batch_size=1000) -> None:
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, motifs: np.ndarray, X_validation, y_validation_e, motifs_validation, y_validation_m):
        num_nodes = X.shape[0]
        nni = torch.tensor(node_node_interactions(X))
        emi_validation = torch.tensor(edge_motif_interactions(X, motifs_validation))

        current_logger = Logger.current_logger()
        self.model = _Node2Vec(X.shape[0], 1, nni, num_nodes)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        epochs = 300
        for epoch in range(epochs):
            self.model.train()

            dataset = MotifIteratorDataset(torch.tensor(X), torch.tensor(motifs))
            training_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            loss_e_sum = 0
            loss_m_sum = 0
            for batch in training_loader:
                X_batch = batch[0][0]
                motifs_batch = batch[1]

                X_, motifs_, y_e, y_m = motif_negative_sampling(X_batch.cpu().detach().numpy(), motifs_batch.cpu().detach().numpy(), 0.5, 1)
                emi_ = torch.tensor(edge_motif_interactions(X_, motifs_))
                y_e = torch.tensor(y_e)
                y_m = torch.tensor(y_m)
                nei = torch.tensor(np.array(X_.nonzero()))

                optimizer.zero_grad()
                y_pred_e, y_pred_m = self.model(nei, emi_)

                loss_e = criterion(y_pred_e, y_e)
                loss_m = criterion(y_pred_m, y_m)

                loss = loss_e

                loss.backward()
                loss_e_sum += loss_e.item()
                loss_m_sum += loss_m.item()
                optimizer.step()
                logging.debug(f"Epoch {epoch} - Loss: {loss.item()}")
            current_logger.report_scalar(title="Loss E", series="N2V Train", iteration=epoch, value=loss_e_sum / len(training_loader))
            current_logger.report_scalar(title="Loss M", series="N2V Train", iteration=epoch, value=loss_m_sum / len(training_loader))

            if epoch % 2 == 0:
                with torch.no_grad():
                    self.model.eval()
                    nei = torch.tensor(np.array(X_validation.nonzero()))                    
                    y_pred_e, y_pred_m = self.model(nei, emi_validation)
                    loss_e = criterion(y_pred_e, torch.tensor(y_validation_e))
                    loss_m = criterion(y_pred_m, torch.tensor(y_validation_m))
                    y_pred_e, y_pred_m = nn.functional.sigmoid(y_pred_e), nn.functional.sigmoid(y_pred_m)
                    current_logger.report_scalar(title="Loss E", series="N2V Validation", iteration=epoch, value=loss_e.item())
                    current_logger.report_scalar(title="Loss M", series="N2V Validation", iteration=epoch, value=loss_m.item())
                    roc_auc_e = roc_auc_score(y_validation_e, y_pred_e.cpu().detach().numpy())
                    roc_auc_m = roc_auc_score(y_validation_m, y_pred_m.cpu().detach().numpy())
                    current_logger.report_scalar(title="ROC AUC", series="N2V Validation", iteration=epoch, value=roc_auc_e)
                    current_logger.report_scalar(title="ROC AUC", series="N2V Validation", iteration=epoch, value=roc_auc_m)
                    logging.debug(f"Validation Loss: {loss.item()}")

    def predict(self, X: np.ndarray, motifs: np.ndarray) -> np.ndarray:
        self.model.eval()
        nei = torch.tensor(np.array(X.nonzero()))
        emi = torch.tensor(edge_motif_interactions(X, motifs))
        y_pred_e, y_pred_m = self.model(nei, emi, sigmoid=True)
        return y_pred_m.cpu().detach().numpy()
    