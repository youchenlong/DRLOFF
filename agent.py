import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import gym
import time
import types


class MLP(nn.Module):
    def __init__(self, n_state, n_action):
        super(MLP, self).__init__()
        self.hid_size = 64
        self.fc1 = nn.Linear(n_state, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
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
    def __init__(self, env):
        super(DQN, self).__init__()
        self.env = env
        self.n_state = self.env.get_state_size()
        self.n_action = self.env.get_action_size()

        # TODO: hyper-parameters should be fine-tuned
        self.batch_size = 128
        self.lr = 0.01
        self.gamma = 0.99
        self.epsilon_start = 0.0
        self.epsilon_finish = 0.99
        self.epsilon_time_length = 10000 # 100 episodes * 100 nodes
        self.epsilon_schedule = LinearSchedule(self.epsilon_start, self.epsilon_finish, self.epsilon_time_length)
        self.buffer_size = 100000 # 1000 episodes * 100 nodes
        self.target_update_interval = 200 # target update interval
        self.grad_norm_clip = 10 # avoid gradient explode

        self.net = MLP(self.n_state, self.n_action)
        self.target_net = MLP(self.n_state, self.n_action)

        self.learn_step_counter = 0
        self.buffer_idx = 0
        # TODO: structure buffer is need to use GNN
        self.buffer = np.zeros((self.buffer_size, self.n_state * 2 + self.n_action + 2))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, state, avail_action, t, evaluate=False):
        if evaluate:
            epsilon = 1.0
        else:
            epsilon = self.epsilon_schedule.eval(t)
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        action_value = self.net.forward(state)
        action_value = action_value.squeeze()
        action_value[avail_action == 0] = 0
        if np.random.randn() <= epsilon:# greedy policy
            action = torch.max(action_value, dim=0)[1].data.numpy()
        else: # random policy
            action = np.random.choice(self.n_action, p=avail_action/sum(avail_action))
            action = action
        return action


    def store_transition(self, state, action, reward, next_state, avail_action):
        transition = np.hstack((state, [action, reward], next_state, avail_action))
        index = self.buffer_idx % self.buffer_size
        self.buffer[index, :] = transition
        self.buffer_idx += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % self.target_update_interval ==0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter+=1

        #sample batch from buffer
        sample_index = np.random.choice(self.buffer_size, self.batch_size)
        batch_memory = self.buffer[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.n_state])
        batch_action = torch.LongTensor(batch_memory[:, self.n_state:self.n_state+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.n_state+1:self.n_state+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, self.n_state+2:self.n_state*2+2])
        batch_avail_action = torch.FloatTensor(batch_memory[:, -self.n_action:])

        q = self.net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_next[batch_avail_action == 0] = 0
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = F.mse_loss(q, q_target)

        self.optimizer.zero_grad()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm_clip)
        if grad_norm > 0:
            print("grad_norm:", grad_norm)
        loss.backward()
        self.optimizer.step()

    def save_models(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(), "{}/net.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.net.load_state_dict(torch.load("{}/net.th".format(path), map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))



# 以下是测试代码

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

def train():
    
    env = gym.make("CartPole-v1")
    def get_state_size(self):
        return self.observation_space.shape[0]
    def get_action_size(self):
        return self.action_space.n
    env.get_state_size = types.MethodType(get_state_size, env)
    env.get_action_size = types.MethodType(get_action_size, env)
    
    agent = DQN(env)
    episodes = 10000
    reward_list = []
    t = 0
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:

            avail_action = np.ones(env.get_action_size())
            action = agent.choose_action(state, avail_action, t)
            next_state, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = next_state
            reward = reward_func(env, x, x_dot, theta, theta_dot)

            agent.store_transition(state, action, reward, next_state, avail_action)            

            ep_reward += reward

            if agent.buffer_idx >= agent.buffer_size:
                agent.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state

            t = t + 1
        reward_list.append(ep_reward)

        if i % 5000 == 0:
            path = "./saved/cartpole-v1/dqn/{}/".format(i)
            agent.save_models(path)
    pass

def evaluate(path=""):
    env = gym.make("CartPole-v1")
    def get_state_size(self):
        return self.observation_space.shape[0]
    def get_action_size(self):
        return self.action_space.n
    env.get_state_size = types.MethodType(get_state_size, env)
    env.get_action_size = types.MethodType(get_action_size, env)
    
    agent = DQN(env)

    if path != "":
        agent.load_models(path)

    state = env.reset()
    ep_reward = 0
    t = 0
    while True:
        time.sleep(0.1)
        env.render()
        avail_action = np.ones(env.get_action_size())
        action = agent.choose_action(state, avail_action, t, True)
        next_state, reward, done, info = env.step(action)

        # x, x_dot, theta, theta_dot = next_state
        # reward = reward_func(env, x, x_dot, theta, theta_dot)            

        ep_reward += reward

        if done:
            print("the episode reward is {}".format(round(ep_reward, 3)))
            break
        state = next_state

        t = t + 1
    pass

if __name__ == "__main__":
    # train()
    evaluate("./saved/cartpole-v1/dqn/")