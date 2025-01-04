import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from utils.scheduler import LinearSchedule
from utils.policy import MLPPolicy
from components.buffer import ReplayBuffer

class MLPAgent:
    def __init__(self, env):
        super(MLPAgent, self).__init__()
        self.env = env
        self.n_state = self.env.get_state_size()
        self.n_action = self.env.get_action_size()

        # TODO: hyper-parameters should be fine-tuned
        self.buffer_size = 500000 # 5000 episodes * 100 nodes
        self.batch_size = 128
        self.lr = 0.01
        self.gamma = 0.99
        self.epsilon_start = 0.0
        self.epsilon_finish = 0.99
        self.epsilon_time_length = 50000 # 500 episodes * 100 nodes
        self.epsilon_schedule = LinearSchedule(self.epsilon_start, self.epsilon_finish, self.epsilon_time_length)
        self.target_update_interval = 5000 # update target network every 50 episodes
        self.grad_norm_clip = 10 # avoid gradient explode

        self.net = MLPPolicy(self.n_state, self.n_action)
        self.target_net = MLPPolicy(self.n_state, self.n_action)

        self.learn_step_counter = 0
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.env)
        self.params = list(self.net.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)


    def choose_action(self, state, avail_action, t=0, evaluate=False):
        if evaluate:
            epsilon = 1.0
        else:
            epsilon = self.epsilon_schedule.eval(t)
        inputs = torch.FloatTensor(self.env.encode_state(state)).unsqueeze(0)
        action_value = self.net.forward(inputs)
        action_value = action_value.squeeze()
        action_value[avail_action == 0] = -9999999
        if np.random.randn() <= epsilon:  # greedy policy
            action = torch.max(action_value, dim=0)[1].item()
        else:  # random policy
            action = int(np.random.choice(self.n_action, p=avail_action/sum(avail_action)))
        if evaluate:
            print(action_value, action)
        return action


    def learn(self):

        #update target parameters
        if self.learn_step_counter % self.target_update_interval ==0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter+=1

        # sample from replay buffer
        batch_state, batch_action, batch_reward, batch_next_state, batch_avail_action, batch_IDs = self.buffer.sample()

        batch_state = torch.FloatTensor(self.env.encode_batch_state(batch_state)) 
        batch_next_state = torch.FloatTensor(self.env.encode_batch_state(batch_next_state))
        batch_action = torch.LongTensor(batch_action.astype(int))
        batch_reward = torch.FloatTensor(batch_reward)
        batch_avail_action = torch.FloatTensor(batch_avail_action)
       
        # calculate loss
        q = torch.gather(self.net(batch_state), dim=1, index=batch_action.unsqueeze(1))
        q_next = self.target_net(batch_next_state).detach()
        q_next[batch_avail_action == 0] = -9999999
        q_target = batch_reward.view(self.batch_size, 1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = F.mse_loss(q, q_target)

        # update parameters
        self.optimizer.zero_grad()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm_clip)
        if grad_norm > 0:
            print("grad_norm:", grad_norm)
        loss.backward()
        self.optimizer.step()


    def store_transition(self, state, action, reward, next_state, avail_action):
        self.buffer.store(state, action, reward, next_state, avail_action)


    def save_models(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(), "{}/net.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))


    def load_models(self, path):
        self.net.load_state_dict(torch.load("{}/net.th".format(path), map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))