import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gnn import GCNLayer

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

        self.gcn_out_dim = 3
        self.hid_size = 128
        self.gcn = GCNLayer(n_feature, self.gcn_out_dim)
        self.fc1 = nn.Linear(num_nodes + num_nodes * self.gcn_out_dim + (M + 2) * 3, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, n_action)

    def forward(self, idx, x, y, adj):
        """
        idx: [batch_size, num_nodes, 1]
        x: [batch_size, num_nodes, 3]
        y: [batch_size, M+2, 3]
        adj: [batch_size, num_nodes, num_nodes]
        """
        batch_size = x.shape[0]
        x = self.gcn(x, adj) # [batch_size, num_nodes, out_dim]
        x = x.reshape(batch_size, -1) # [batch_size, num_nodes * out_dim]
        idx = idx.reshape(batch_size, -1) # [batch_size, num_nodes * 1]
        y = y.reshape(batch_size, -1) # [batch_size, (M+2) * 3]
        x = F.relu(self.fc1(torch.cat([idx, x, y], dim=-1)))
        action_prob = self.fc2(x)
        return action_prob