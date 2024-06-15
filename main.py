import time
from environment import Environment
from agent import DQN
from baselines import LocalAgent, EdgeAgent, CloudAgent, RandomAgent, GreedyAgent

def main(name="greedy", seed=0):
    env = Environment()

    if name == "local":
        agent = LocalAgent()
    elif name == "edge":
        agent = EdgeAgent()
    elif name == "cloud":
        agent = CloudAgent()
    elif name == "random":
        agent = RandomAgent()
    elif name == "greedy":
        agent = GreedyAgent()
    elif name == "dqn":
        agent = DQN(env)
        path = ""
        if path != "":
            agent.load_models(path)
    
    state = env.reset(seed)
    ep_reward = 0
    while not env.done:
        avail_action = env.get_avail_actions()
        action = agent.choose_action(state, avail_action)

        state, reward, done = env.step(action)

        ep_reward = ep_reward + reward
    # env.log()
    # print(env.get_metric())
    # env.plot_task()
    return env.get_metric(), ep_reward

if __name__ == "__main__":
    start_time = time.time()
    # start_time = 0
    for name in ["local", "edge", "cloud", "random", "greedy", "dqn"]:
        episodes = 200
        dvr_rate_mean = 0
        ep_reward_mean = 0
        for i in range(episodes):
            dvr_rate, ep_reward = main(name, int(start_time)+i)
            dvr_rate_mean += dvr_rate / episodes
            ep_reward_mean += ep_reward / episodes
        print("agent: {}\t, dvr_rate_mean: {}\t\t, ep_reward_mean: {}".format(name, dvr_rate_mean, ep_reward_mean))