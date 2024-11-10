import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from environment import Environment
from buffer import ReplayBuffer
from gnn import GCNLayer


class MLPPolicy(nn.Module):
    def __init__(self, n_state, n_action):
        super(MLPPolicy, self).__init__()
        self.name = "mlp"

        self.hid_size = 128
        self.fc1 = nn.Linear(n_state, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, n_action)

    def forward(self, x):
        """
        x: [batch_size, n_state]
        n_state = num_nodes * 4 + (M+2)*3 + num_nodes * num_nodes
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


class LinearSchedule():
    def __init__(self,
                 start,
                 finish,
                 time_length):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.finish - self.start) / self.time_length

    def eval(self, T):
        return min(self.finish, self.start + self.delta * T)

class DQN():
    def __init__(self, env, name="mlp"):
        super(DQN, self).__init__()
        self.env = env
        self.n_state = self.env.get_state_size()
        self.n_action = self.env.get_action_size()

        # TODO: hyper-parameters should be fine-tuned
        self.buffer_size = 100000 # 1000 episodes * 100 nodes
        self.batch_size = 128
        self.lr = 0.01
        self.gamma = 0.99
        self.epsilon_start = 0.0
        self.epsilon_finish = 0.99
        self.epsilon_time_length = 10000 # 100 episodes * 100 nodes
        self.epsilon_schedule = LinearSchedule(self.epsilon_start, self.epsilon_finish, self.epsilon_time_length)
        self.target_update_interval = 200 # target update interval
        self.grad_norm_clip = 10 # avoid gradient explode

        if name == "mlp":
            self.net = MLPPolicy(self.n_state, self.n_action)
            self.target_net = MLPPolicy(self.n_state, self.n_action)
        elif name == "gcn":
            self.net = GCNPolicy(3, self.n_action, self.env.max_num_nodes, self.env.M)
            self.target_net = GCNPolicy(3, self.n_action, self.env.max_num_nodes, self.env.M)
        else:
            raise Exception("error!")

        self.learn_step_counter = 0
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.env)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)


    def choose_action(self, state, avail_action, t=0, evaluate=False):
        if evaluate:
            epsilon = 1.0
        else:
            epsilon = self.epsilon_schedule.eval(t)
        if self.net.name == "mlp":
            state = self.env.encode_state(state)
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action_value = self.net.forward(state)
        elif self.net.name == "gcn":
            task_idx, task_info, dev_info = state
            adj = self.env.adjs[self.env.ID]
            task_idx = torch.unsqueeze(torch.FloatTensor(task_idx), 0)
            task_info = torch.unsqueeze(torch.FloatTensor(task_info), 0)
            dev_info = torch.unsqueeze(torch.FloatTensor(dev_info), 0)
            adj = torch.unsqueeze(torch.FloatTensor(adj), 0) 
            action_value = self.net.forward(task_idx, task_info, dev_info, adj)
        else:
            raise Exception("error!")
            
        action_value = action_value.squeeze()
        action_value[avail_action == 0] = 0
        if np.random.randn() <= epsilon:  # greedy policy
            action = torch.max(action_value, dim=0)[1].data.numpy()
        else:  # random policy
            action = np.random.choice(self.n_action, p=avail_action/sum(avail_action))
        return action


    def learn(self):

        #update target parameters
        if self.learn_step_counter % self.target_update_interval ==0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter+=1

        # sample from replay buffer
        batch_state, batch_action, batch_reward, batch_next_state, batch_avail_action, batch_IDs = self.buffer.sample()
        if self.net.name == "mlp":
            batch_state = torch.FloatTensor(self.env.encode_batch_state(batch_state)) 
            batch_next_state = torch.FloatTensor(self.env.encode_batch_state(batch_next_state))
            batch_action = torch.LongTensor(batch_action.astype(int))
            batch_reward = torch.FloatTensor(batch_reward)
            batch_avail_action = torch.FloatTensor(batch_avail_action)
            q = torch.gather(self.net(batch_state), dim=1, index=batch_action.unsqueeze(1))
            q_next = self.target_net(batch_next_state).detach()
            q_next[batch_avail_action == 0] = 0
            q_target = batch_reward.view(self.batch_size, 1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        elif self.net.name == "gcn":
            idx, x, y = batch_state
            target_idx, target_x, target_y = batch_next_state
            adj = np.array([self.env.adjs[ID] for ID in batch_IDs])
            batch_action = torch.LongTensor(batch_action.astype(int))
            batch_reward = torch.FloatTensor(batch_reward)
            batch_avail_action = torch.FloatTensor(batch_avail_action)
            q = torch.gather(self.net(torch.FloatTensor(idx), torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(adj)), dim=1, index=batch_action.unsqueeze(1))
            q_next = self.target_net(torch.FloatTensor(target_idx), torch.FloatTensor(target_x), torch.FloatTensor(target_y), torch.FloatTensor(adj)).detach()
            q_next[batch_avail_action == 0] = 0
            q_target = batch_reward.view(self.batch_size, 1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        else:
            raise Exception("error!")

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


def save(dirname, filename, data):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(os.path.join(dirname, filename), 'w') as f:
        f.write(str(data))

def train():
    start_time = time.time()
    env = Environment()
    agent = DQN(env, name="gcn")
    episodes = 50000
    dvr_list = []
    reward_list = []
    t = 0
    for i in range(episodes):
        state = env.reset(seed=int(start_time)+i)
        ep_reward = 0
        done = False
        while not done:
            avail_action = env.get_avail_actions()
            action = agent.choose_action(state, avail_action, t)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, avail_action)            

            ep_reward += reward

            if agent.buffer.can_sample():
                agent.learn()                    
            if done:
                print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                break
            state = next_state
            t = t + 1
        dvr_rate = env.get_metric()
        dvr_list.append(dvr_rate)
        reward_list.append(ep_reward)

        if i % 1000 == 0:
            env.update_adjs(set(agent.buffer.IDs))

        if i % 1000 == 0:
            agent.save_models("./saved/off/dqn/{}/{}".format(start_time, i))
            save("./saved/off/dqn/{}".format(start_time), "dvr.txt", dvr_list)
            save("./saved/off/dqn/{}".format(start_time), "ep_reward.txt", reward_list)
        

if __name__ == '__main__':
    train()