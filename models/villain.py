import torch
import numpy as np
from models.geonvillain.model import model
from utils import CustomEstimator
from motif import motif_negative_sampling, edge_motif_interactions, node_node_interactions
from torch import nn
import math
import torch_geometric.nn.aggr
import copy

class VilLain(CustomEstimator):

    def fit(self, X: np.ndarray, motifs: np.ndarray, X_validation, y_validation_e, motifs_validation, y_validation_m):
        dim = 128
        num_labels = 2
        num_step=4
        num_step_gen=100
        lr=0.01
        epochs = 500
        num_subspace = math.ceil(dim / num_labels)
        nei = torch.tensor(np.array(X.nonzero()))
        V_idx = nei[0]
        E_idx = nei[1]
        self.aggr_1 = torch_geometric.nn.aggr.MaxAggregation()
        self.aggr_2 = torch_geometric.nn.aggr.MinAggregation()
        V, E = torch.max(V_idx)+1, torch.max(E_idx)+1
        self.model = model(V_idx, E_idx, V, E, num_subspace, num_labels, num_step, num_step_gen)
        self.linear = nn.Linear(dim, 1)
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

            if epoch % 10 == 0:
                print('{}\t{}\t{}'.format(epoch, loss_local.item(), loss_global.item()))
            
            if epoch <= 100:
                continue

            if loss.item() < best_loss:
                self.model.eval()
                best_loss = loss.item()
                best_model = copy.deepcopy(self.model.state_dict())

            diff = abs(loss.item() - pre_loss) / abs(pre_loss)
            if diff < 0.002:
                cnt_wait += 1
            else:
                cnt_wait = 0
                    
            if cnt_wait == patience:
                break
                
            pre_loss = loss.item()
       
        self.model.load_state_dict(best_model)

        optimizer_linear = torch.optim.AdamW(self.linear.parameters(), lr=lr, weight_decay=0)
        

    def predict(self, X: np.ndarray, motifs: np.ndarray):
        self.model.eval()
        self.linear.eval()
        node_embeds = self.model.get_node_embeds()
        local_loss, global_loss = self.model()
        nei = torch.tensor(np.array(X.nonzero()))
        edge_embeds = self.aggr_1(node_embeds[nei[0]], nei[1])
        edge_embeds = self.linear(edge_embeds)
        y_e = nn.functional.sigmoid(edge_embeds)
        y_e = y_e.cpu().detach().numpy().reshape(-1, 1)
        y_m = y_e[motifs].prod(axis=1)
        return y_m
