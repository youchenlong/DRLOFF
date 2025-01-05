import numpy as np
import time
import os

from env.environment import Environment
from components.episodebuffer import Episode, ReplayBuffer
from modules.agents.gcnagent import GCNAgent

def save(dirname, filename, data):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(os.path.join(dirname, filename), 'w') as f:
        f.write(str(data))

def train():
    start_time = time.time()
    env = Environment()
    agent = GCNAgent(env)
    # episodes = 50001
    episodes = 5001
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
            agent.save_models("./saved/off/gcn/{}/{}".format(start_time, i))
            save("./saved/off/gcn/{}".format(start_time), "dvr.txt", dvr_list)
            save("./saved/off/gcn/{}".format(start_time), "ep_reward.txt", reward_list)
        

if __name__ == '__main__':
    train()