import torch
import torch.nn as nn
import torch.nn.functional as F
# pip install torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

class MLPPolicy(nn.Module):
    def __init__(self, input_shape, n_action):
        super(MLPPolicy, self).__init__()
        self.name = "mlp"

        self.hid_size = 128
        self.fc1 = nn.Linear(input_shape, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, n_action)

    def forward(self, x):
        """
        x: [batch_size, input_shape]
        input_shape = n_state + character_dim + mental_dim
        """
        x = F.relu(self.fc1(x))
        action_prob = self.fc2(x)
        return action_prob


class GCNPolicy(nn.Module):
    def __init__(self, n_feature, n_action, num_nodes, M):
        super(GCNPolicy, self).__init__()
        self.name = "gcn"

        self.gnn_hid_size = 3
        self.gnn_out_dim = 3
        self.hid_size = 128
        self.gcn1 = GCNConv(n_feature, self.gnn_hid_size)
        self.gcn2 = GCNConv(self.gnn_hid_size, self.gnn_out_dim)
        self.fc1 = nn.Linear(num_nodes + num_nodes * self.gnn_out_dim + (M + 2) * 3, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, n_action)

    def forward(self, idx, x, y, adj):
        """
        idx: [batch_size, num_nodes]
        x: [batch_size, num_nodes, 3]
        y: [batch_size, M+2, 3]
        adj: [batch_size, num_nodes, num_nodes]
        """
        batch_size = x.shape[0]

        data_list = [Data(x=x[i], edge_index=dense_to_sparse(adj[i])[0]) for i in range(batch_size)]
        batch = Batch.from_data_list(data_list)
        x = F.relu(self.gcn1(batch.x, batch.edge_index)) # [batch_size * num_nodes, gnn_hid_size]
        x = self.gcn2(x, batch.edge_index) # [batch_size * num_nodes, gnn_out_dim]

        x = x.reshape(batch_size, -1) # [batch_size, num_nodes * gnn_out_dim]
        idx = idx.reshape(batch_size, -1) # [batch_size, num_nodes * 1]
        y = y.reshape(batch_size, -1) # [batch_size, (M+2) * 3]
        x = F.relu(self.fc1(torch.cat([idx, x, y], dim=-1)))
        action_prob = self.fc2(x)
        return action_prob


class GATPolicy(nn.Module):
    def __init__(self, n_feature, n_action, num_nodes, M):
        super(GATPolicy, self).__init__()
        self.name = "gat"

        self.heads = 2
        self.gnn_hid_size = 3
        self.gnn_out_dim = 3
        self.hid_size = 128
        self.gat1 = GATConv(n_feature, self.gnn_hid_size, heads=self.heads)
        self.gat2 = GATConv(self.gnn_hid_size * self.heads, self.gnn_out_dim)
        self.fc1 = nn.Linear(num_nodes + num_nodes * self.gnn_out_dim + (M + 2) * 3, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, n_action)

    def forward(self, idx, x, y, adj):
        """
        idx: [batch_size, num_nodes]
        x: [batch_size, num_nodes, 3]
        y: [batch_size, M+2, 3]
        adj: [batch_size, num_nodes, num_nodes]
        """
        batch_size = x.shape[0]

        data_list = [Data(x=x[i], edge_index=dense_to_sparse(adj[i])[0]) for i in range(batch_size)]
        batch = Batch.from_data_list(data_list)
        x = F.relu(self.gat1(batch.x, batch.edge_index)) # [batch_size * num_nodes, gnn_hid_size * heads]
        x = self.gat2(x, batch.edge_index) # [batch_size * num_nodes, gnn_out_dim]

        x = x.reshape(batch_size, -1) # [batch_size, num_nodes * gnn_out_dim]
        idx = idx.reshape(batch_size, -1) # [batch_size, num_nodes * 1]
        y = y.reshape(batch_size, -1) # [batch_size, (M+2) * 3]
        x = F.relu(self.fc1(torch.cat([idx, x, y], dim=-1)))
        action_prob = self.fc2(x)
        return action_prob