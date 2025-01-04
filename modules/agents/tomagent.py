import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from utils.scheduler import LinearSchedule
from utils.policy import MLPPolicy
from components.episodebuffer import ReplayBuffer
from modules.tom.observer import Observer

class ToMAgent:
    def __init__(self, env):
        super(ToMAgent, self).__init__()
        self.env = env
        self.n_state = self.env.get_state_size()
        self.n_action = self.env.get_action_size()
        self.observer = Observer(env)
        self.character_dim = self.observer.cnet.character_dim
        self.mental_dim = self.observer.mnet.mental_dim

        # TODO: hyper-parameters should be fine-tuned
        self.buffer_size = 5000 # 5000 episodes
        self.batch_size = 16
        self.lr = 0.01
        self.gamma = 0.99
        self.epsilon_start = 0.0
        self.epsilon_finish = 0.99
        self.epsilon_time_length = 50000 # 100 episodes * 100 nodes
        self.epsilon_schedule = LinearSchedule(self.epsilon_start, self.epsilon_finish, self.epsilon_time_length)
        self.target_update_interval = 50 # update target network every 50 episodes
        self.grad_norm_clip = 10 # avoid gradient explode

        self.net = MLPPolicy(self.n_state + self.character_dim + self.mental_dim, self.n_action)
        self.target_net = MLPPolicy(self.n_state + self.character_dim + self.mental_dim, self.n_action)

        self.learn_step_counter = 0
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.params = list(self.net.parameters()) + list(self.observer.cnet.parameters()) + list(self.observer.mnet.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        self.e_character = None
        self.e_mental = None
        self.hidden_state = None


    def choose_action(self, state, avail_action, t=0, evaluate=False):
        if self.e_character is None:
            self.e_character = torch.zeros(1, self.observer.cnet.character_dim)
        if self.e_mental is None:
            self.e_mental = torch.zeros(1, self.observer.mnet.mental_dim)
        if evaluate:
            epsilon = 1.0
        else:
            epsilon = self.epsilon_schedule.eval(t)

        inputs = torch.cat([torch.FloatTensor(self.env.encode_state(state)).unsqueeze(0), self.e_character, self.e_mental], dim=-1)
        action_value = self.net.forward(inputs)
        action_value = action_value.squeeze()
        action_value[avail_action == 0] = 0
        if np.random.randn() <= epsilon:  # greedy policy
            action = torch.max(action_value, dim=0)[1].item()
        else:  # random policy
            action = int(np.random.choice(self.n_action, p=avail_action/sum(avail_action)))
        
        # calculate mental embedding
        self.e_mental, self.hidden_state = self.observer.calc_mental(state, action, self.e_character, self.hidden_state)

        return action


    def learn(self):

        #update target parameters
        if self.learn_step_counter % self.target_update_interval ==0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter+=1

        # sample from replay buffer
        episodes = self.buffer.sample()

        # calculate character embedding
        self.e_character = self.observer.calc_character(episodes)
        
        # get relevant quantities
        states, actions, rewards, next_states, avail_actions = [], [], [], [], []
        e_mentals, next_e_mentals = [], []
        self.init_hidden()
        for episode in episodes:
            timesteps = len(episode.states)
            for t in range(timesteps):
                self.e_mental, self.hidden_state = self.observer.calc_mental(episode.states[t], episode.actions[t], self.e_character, self.hidden_state)            
                if t < timesteps - 1:
                    states.append(self.env.encode_state(episode.states[t]))
                    actions.append(episode.actions[t])
                    rewards.append(episode.rewards[t])
                    next_states.append(self.env.encode_state(episode.next_states[t]))
                    avail_actions.append(episode.avail_actions[t])
                    e_mentals.append(self.e_mental.squeeze())
                if t > 0:
                    next_e_mentals.append(self.e_mental.squeeze())
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        avail_actions = torch.FloatTensor(np.array(avail_actions))
        e_mentals = torch.stack(e_mentals, dim=0)
        e_characters = self.e_character.expand_as(e_mentals)

        # calculate loss
        inputs = torch.cat([states, e_characters, e_mentals], dim=1)
        next_inputs = torch.cat([next_states, e_characters, e_mentals], dim=1)
        q = torch.gather(self.net(inputs), dim=1, index=actions.unsqueeze(1))
        q_next = self.target_net(next_inputs).detach()
        q_next[avail_actions == 0] = 0
        q_target = (rewards + self.gamma * q_next.max(1)[0]).unsqueeze(1)       
        loss = F.mse_loss(q, q_target)

        # update parameters
        self.optimizer.zero_grad()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm_clip)
        if grad_norm > 0:
            print("grad_norm:", grad_norm)
        loss.backward()
        self.optimizer.step()


    def init_hidden(self):
        self.hidden_state = self.observer.mnet.init_hidden()


    def store_episode(self, episode):
        self.buffer.insert_an_episode(episode)
    

    def save_models(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(), "{}/net.th".format(path))
        torch.save(self.observer.cnet.state_dict(), "{}/cnet.th".format(path))
        torch.save(self.observer.mnet.state_dict(), "{}/mnet.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))


    def load_models(self, path):
        self.net.load_state_dict(torch.load("{}/net.th".format(path), map_location=lambda storage, loc: storage))
        self.observer.cnet.load_state_dict(torch.load("{}/cnet.th".format(path), map_location=lambda storage, loc: storage))
        self.observer.mnet.load_state_dict(torch.load("{}/mnet.th".format(path), map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))