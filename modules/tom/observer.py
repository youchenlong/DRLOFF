import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CharacterNetwork(nn.Module):
    def __init__(self, input_shape):
        super(CharacterNetwork, self).__init__()
        self.input_shape = input_shape
        self.rnn_hidden_dim = 32
        self.character_dim = 8

        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.character_dim)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()


    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        hidden_state = self.rnn(x, hidden_state)
        e_character = self.fc2(hidden_state)
        return e_character, hidden_state


class MentalNetwork(nn.Module):
    def __init__(self, input_shape):
        super(MentalNetwork, self).__init__()
        self.input_shape = input_shape
        self.rnn_hidden_dim = 32
        self.mental_dim = 8

        self.fc1 = nn.Linear(self.input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.mental_dim)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()


    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        hidden_state = self.rnn(x, hidden_state)
        e_mental = self.fc2(hidden_state)
        return e_mental, hidden_state


class Observer:
    def __init__(self, env):
        self.env = env
        self.n_state = env.get_state_size()
        self.n_action = env.get_action_size()

        self.cnet = CharacterNetwork(self.n_state + self.n_action)
        self.mnet = MentalNetwork(self.n_state + self.n_action + self.cnet.character_dim)


    def calc_character(self, episodes):
        e_character_sum = torch.zeros(1, self.cnet.character_dim)
        for episode in episodes:
            timesteps = len(episode.states)
            hidden_state = self.cnet.init_hidden()
            for t in range(timesteps):
                state, action = episode.states[t], episode.actions[t]
                state = torch.FloatTensor(self.env.encode_state(state)).unsqueeze(0)
                action_onehot = torch.LongTensor(np.eye(self.n_action)[action]).unsqueeze(0)
                inputs = torch.cat([state, action_onehot], dim=1)
                e_character, hidden_state = self.cnet(inputs, hidden_state)
            e_character_sum += e_character
        return e_character_sum


    def calc_mental(self, state, action, e_character, hidden_state):
        if e_character is None:
            e_character = torch.zeros(1, self.cnet.character_dim)
        state = torch.FloatTensor(self.env.encode_state(state)).unsqueeze(0)
        action_onehot = torch.LongTensor(np.eye(self.n_action)[action]).unsqueeze(0)
        inputs = torch.cat([state, action_onehot, e_character], dim=1)
        e_mental, hidden_state = self.mnet(inputs, hidden_state)
        return e_mental, hidden_state