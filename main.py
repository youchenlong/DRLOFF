from environment import Environment
from agent import DQN

def save(dirname, filename, data):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(os.path.join(dirname, filename), 'w') as f:
        f.write(data)

def train():
    env = Environment()
    agent = DQN(env)
    episodes = 20000
    dvr_list = []
    reward_list = []
    t = 0
    for i in range(episodes):
        # print(i)
        state = env.reset(seed=i)
        ep_reward = 0
        done = False
        while not done:
            avail_action = env.get_avail_actions()
            action = agent.choose_action(state, avail_action, t)
            next_state, reward, done = env.step(action)

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
        
        dvr_rate = env.get_metric()
        dvr_list.append(dvr_rate)
        reward_list.append(ep_reward)

        if i % 5000 == 0:
            agent.save_models("./saved/off/dqn/{}/".format(i))
    save("./saved/off/dqn/", "dvr.txt", dvr_list)
    save("./saved/off/dqn/", "ep_reward.txt", reward_list)


def evaluate(path=""):
    env = Environment()
    
    agent = DQN(env)

    if path != "":
        agent.load_models(path)

    state = env.reset()
    ep_reward = 0
    t = 0
    while True:

        avail_action = env.get_avail_actions()
        action = agent.choose_action(state, avail_action, t, True)
        next_state, reward, done = env.step(action)     

        ep_reward += reward

        if done:
            print("the episode reward is {}".format(round(ep_reward, 3)))
            break
        state = next_state

        t = t + 1
    pass
        

if __name__ == '__main__':
    train()
    # evaluate()