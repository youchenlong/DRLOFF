import numpy as np
import time
import os

from env.environment import Environment
from components.episodebuffer import Episode, ReplayBuffer
from modules.agents.tomagent import ToMAgent

def save(dirname, filename, data):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(os.path.join(dirname, filename), 'w') as f:
        f.write(str(data))

def main():
    start_time = time.time()
    env = Environment()
    agent = ToMAgent(env)
    # episodes = 50001
    episodes = 5001
    dvr_list = []
    reward_list = []
    t = 0
    for i in range(episodes):
        episode = Episode(env)
        states, actions, rewards, next_states, avail_actions = [], [], [], [], []
        agent.init_hidden()

        # sample
        state = env.reset(seed=int(start_time)+i)
        ep_reward = 0
        done = False
        while not done:
            # select action
            avail_action = env.get_avail_actions()
            action = agent.choose_action(state, avail_action, t)

            # step
            next_state, reward, done = env.step(action)  

            ep_reward += reward 
            
            if done:
                print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                break

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            avail_actions.append(avail_action)  
            state = next_state
            t = t + 1

        # store
        episode.update(states, actions, rewards, next_states, avail_actions, env.ID)
        agent.store_episode(episode)

        # learn
        if agent.buffer.can_sample():
            agent.learn()
        
        dvr_rate = env.get_metric()
        dvr_list.append(dvr_rate)
        reward_list.append(ep_reward)

        if i % 500 == 0:
            env.update_adjs(set(agent.buffer.get_IDs()))

        if i % 500 == 0:
            agent.save_models("./saved/off/tom/{}/{}".format(start_time, i))
            save("./saved/off/tom/{}".format(start_time), "dvr.txt", dvr_list)
            save("./saved/off/tom/{}".format(start_time), "ep_reward.txt", reward_list)
        

if __name__ == '__main__':
    main()