import numpy as np

from torch import nn
import torch_geometric.nn.aggr
from torch_geometric.nn import Node2Vec, HypergraphConv, GCNConv

class Node2VecHypergraphConv(nn.Module): # Da testare se da le stesse performance

    def __init__(self, channels, edge_index) -> None:
        super().__init__()
        self.node2vec = Node2Vec(edge_index, embedding_dim=channels, walk_length=80, context_size=80, walks_per_node=10, num_negative_samples=1, p=0.5, q=2, sparse=False)
        self.hypergraph_node_conv = HypergraphConv(channels, channels, attention_mode="edge")
        self.linear = nn.Linear(channels, channels)

    def forward(self, edge_index):
        y = self.node2vec()
        y = self.hypergraph_node_conv(y, edge_index)
        y = nn.functional.leaky_relu(y)
        y = self.linear(y)
        return y

class HypergraphMotifConvE(nn.Module):

    def __init__(self, channels_1, channels_2, channels_3, channels_out, edge_index) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.5)

        self.node2vec_hypergraph_conv = Node2VecHypergraphConv(channels_1, edge_index)

        self.aggr_1 = torch_geometric.nn.MeanAggregation()
        self.hypergraph_edge_conv_1 = GCNConv(channels_1, channels_2)
        self.linear_2 = nn.Linear(channels_2, channels_2)
        self.edge_linear_out = nn.Linear(channels_2, channels_out)

        self.aggr_2 = torch_geometric.nn.aggr.MaxAggregation()
        self.linear_3 = nn.Linear(channels_2, channels_3)

        self.linear_out = nn.Linear(channels_3, channels_out)
    
    def node_embeddings(self, edge_index):
        y = self.node2vec_hypergraph_conv(edge_index)
        return y
    
    def edge_embeddings(self, X, edge_index, edge_edge_index, sigmoid=False):
        y = X
        y = y.T @ y
        y = nn.functional.relu(y)
        y = self.aggr_1(y[edge_index[0]], edge_index[1])
        y = self.dropout(y)
        y = self.hypergraph_edge_conv_1(y, edge_edge_index)
        y = nn.functional.relu(y)
        y = self.linear_2(y)
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

import torch
from utils import CustomEstimator
from motif import motif_negative_sampling, edge_motif_interactions, node_node_interactions
import logging

from time import time

class HypergraphMotifConv(CustomEstimator):

    def fit(self, X: np.ndarray, motifs: np.ndarray):
        nni = torch.tensor(node_node_interactions(X))
        eei = torch.tensor(node_node_interactions(X.T, 3))

        self.eei = eei

        self.model = HypergraphMotifConvE(X.shape[0], 64, 64, 1, nni)
        self.model.train()

        epochs = 100
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            
            X_, motifs_, y_e, y_m = motif_negative_sampling(X, motifs, 0.5, 1)
            y_m = torch.tensor(y_m)
            nei = torch.tensor(np.array(X_.nonzero()))
            emi = torch.tensor(edge_motif_interactions(X_, motifs_))

            optimizer.zero_grad()
            y = self.model.node_embeddings(nei)
            y, y_pred_e = self.model.edge_embeddings(y, nei, eei)
            _, y_pred_m = self.model.motif_embeddings(y, emi)
            loss = criterion(y_pred_m, y_m)
            loss.backward()
            optimizer.step()
            logging.debug(f"Epoch {epoch} - Loss: {loss.item()}")

    def predict(self, X: np.ndarray, motifs: np.ndarray):
        self.model.eval()
        nei = torch.tensor(np.array(X.nonzero()))
        eei = self.eei
        emi = torch.tensor(edge_motif_interactions(X, motifs))

        y = self.model.node_embeddings(nei)
        y, y_pred_e = self.model.edge_embeddings(y, nei, eei)
        _, y_pred_m = self.model.motif_embeddings(y, emi, sigmoid=True)
        y_pred_m = y_pred_m.cpu().detach().numpy()
        return y_pred_m
