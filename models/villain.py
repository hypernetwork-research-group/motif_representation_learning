import torch
import numpy as np
from models.geonvillain.model import model
from utils import CustomEstimator
from motif import motif_negative_sampling, edge_motif_interactions, node_node_interactions
from torch import nn
import math
import torch_geometric.nn.aggr
import copy
from utils import MotifIteratorDataset
from torch.utils.data import DataLoader
from clearml import Logger
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear_1 = nn.Linear(in_features, in_features // 2)
        self.linear_2 = nn.Linear(in_features // 2, out_features)
        self.aggr_1 = torch_geometric.nn.aggr.MaxAggregation()
        self.aggr_2 = torch_geometric.nn.aggr.MulAggregation()

    def forward(self, X: torch.Tensor, nei, emi, sigmoid=False) -> torch.Tensor:
        # Initial dropout
        y = self.dropout(X)
        # Edges
        y_e = self.aggr_1(y[nei[0]], nei[1])
        y_e = self.linear_1(y_e)
        y_e = nn.functional.leaky_relu(y_e)
        y_e = self.linear_2(y_e)
        # Motifs
        y_m = self.aggr_2(y_e[emi[0]], emi[1])
        # Out
        if sigmoid:
            y_e = nn.functional.sigmoid(y_e)
            y_m = nn.functional.sigmoid(y_m)
        return y_e, y_m

class VilLain(CustomEstimator):

    def __init__(self, n_features):
        self.n_features = n_features

    def fit(self, X: np.ndarray):
        missing_nodes = np.where(np.sum(X, axis=1) == 0)[0]
        # Add singleton edge for missing nodes
        X_ = np.hstack((X, np.eye(X.shape[0])[missing_nodes].T))
        print(X_.shape)
        self.node_embeds = np.empty((X_.shape[0], 0))
        for num_labels in [2, 3, 4, 5, 6, 7, 8]:
            dim = math.ceil(X_.shape[0] / 7) # Verificare se funziona
            num_step=4
            num_step_gen=100
            lr=0.01
            epochs = 5000
            num_subspace = math.ceil(dim / num_labels)
            nei = torch.tensor(np.array(X_.nonzero()))
            V_idx = nei[0]
            E_idx = nei[1]
            V, E = torch.max(V_idx) + 1, torch.max(E_idx) + 1
            self.model = model(V_idx, E_idx, V, E, num_subspace, num_labels, num_step, num_step_gen)
            best_loss, best_model = 1e10, None
            pre_loss, patience = 1e10, 20

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0)

            for epoch in range(1, epochs + 1):
                self.model.train()

                loss_local, loss_global = self.model()
                loss = loss_local + loss_global

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                # if epoch % 10 == 0:
                    # print('{}\t{}\t{}'.format(epoch, loss_local.item(), loss_global.item()))
                
                if epoch <= 100:
                    continue

                if loss.item() < best_loss:
                    self.model.eval()
                    best_loss = loss.item()
                    best_model = copy.deepcopy(self.model.state_dict())

                diff = abs(loss.item() - pre_loss + 1e-10) / abs(pre_loss + 1e-10)
                if diff < 0.002:
                    cnt_wait += 1
                else:
                    cnt_wait = 0

                if cnt_wait == patience:
                    break

                pre_loss = loss.item()

            self.model.load_state_dict(best_model)
            self.model.eval()
            node_embeds = np.array(self.model.get_node_embeds())
            self.node_embeds = np.concatenate((self.node_embeds, node_embeds), axis=1)
        print(self.node_embeds.shape, self.n_features, X_.shape[0])
        pca = PCA(n_components=self.n_features)
        self.node_embeds = pca.fit_transform(self.node_embeds)
        self.node_embeds = self.node_embeds.astype(np.float32)

class VilLainSLP(CustomEstimator):

    def fit(self, X: np.ndarray, motifs: np.ndarray, X_validation, y_validation_e, motifs_validation, y_validation_m):
        current_logger = Logger.current_logger()
        self.villain = VilLain(X.shape[0])
        self.villain.fit(X)

        node_embeds = torch.tensor(self.villain.node_embeds)

        nei_validation = torch.tensor(np.array(X_validation.nonzero()))
        emi_validation = torch.tensor(edge_motif_interactions(X_validation, motifs_validation))

        epochs = 200
        self.linear = Linear(X.shape[0], 1)
        optimizer_linear = torch.optim.Adam(self.linear.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            self.linear.train()

            dataset = MotifIteratorDataset(torch.tensor(X), torch.tensor(motifs))
            training_loader = DataLoader(dataset, batch_size=1000, shuffle=True)

            loss_e_sum = 0
            loss_m_sum = 0
            for batch in training_loader:
                X_batch = batch[0][0]
                motifs_batch = batch[1]

                X_, motifs_, y_e, y_m = motif_negative_sampling(X_batch.cpu().detach().numpy(), motifs_batch.cpu().detach().numpy(), 0.5, 1)
                nei = torch.tensor(np.array(X_.nonzero()))
                emi = torch.tensor(edge_motif_interactions(X_, motifs_))
                y_e = torch.tensor(y_e)
                y_m = torch.tensor(y_m)

                optimizer_linear.zero_grad()

                y_pred_e, y_pred_m = self.linear(node_embeds, nei, emi)

                loss_e = criterion(y_pred_e, y_e)
                loss_m = criterion(y_pred_m, y_m)

                loss = loss_e + loss_m

                loss.backward()

                loss_e_sum += loss_e.item()
                loss_m_sum += loss_m.item()
                optimizer_linear.step()

            current_logger.report_scalar("Loss E", "ViLlain Train", iteration = epoch, value = loss_e_sum / len(training_loader))
            current_logger.report_scalar("Loss M", "ViLlain Train", iteration = epoch, value = loss_m_sum / len(training_loader))

            if epoch % 2 == 0:
                with torch.no_grad():
                    self.linear.eval()
                    y_pred_e, y_pred_m = self.linear(node_embeds, nei_validation, emi_validation, sigmoid=True)
                    loss_e = criterion(y_pred_e, torch.tensor(y_validation_e))
                    loss_m = criterion(torch.tensor(y_validation_m), y_pred_m)
                    current_logger.report_scalar("Loss M", "ViLlain Validation", iteration = epoch, value = loss_m.item())
                    current_logger.report_scalar("Loss E", "ViLlain Validation", iteration = epoch, value = loss_m.item())
                    auc_e = roc_auc_score(y_validation_e, y_pred_e)
                    current_logger.report_scalar("ROC AUC E", "ViLlain Validation", iteration = epoch, value = auc_e)
                    auc_m = roc_auc_score(y_validation_m, y_pred_m)
                    current_logger.report_scalar("ROC AUC", "ViLlain Validation", iteration = epoch, value = auc_m)

    def predict(self, X: np.ndarray, motifs: np.ndarray):
        self.linear.eval()
        node_embeds = torch.tensor(self.villain.node_embeds)
        nei = torch.tensor(np.array(X.nonzero()))
        emi = torch.tensor(edge_motif_interactions(X, motifs))
        y_pred_e, y_m = self.linear(node_embeds, nei, emi, sigmoid=True)
        y_m = y_m.cpu().detach().numpy()
        return y_m
