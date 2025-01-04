import time
from env.environment import Environment
from modules.agents.mlpagent import MLPAgent
from modules.agents.gcnagent import GCNAgent
from modules.agents.tomagent import ToMAgent
from modules.agents.baselines import LocalAgent, EdgeAgent, CloudAgent, RandomAgent, GreedyAgent

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
    elif name == "mlp":
        agent = MLPAgent(env)
        # TODO: set your model path
        path = ""
        if path != "":
            agent.load_models(path)
    elif name == "gcn":
        agent = GCNAgent(env)
        path = ""
        if path != "":
            agent.load_models(path)
    elif name == "tom":
        agent = ToMAgent(env)
        path = ""
        if path != "":
            agent.load_models(path)
        agent.init_hidden()
    
    state = env.reset(seed)
    ep_reward = 0
    while not env.done:
        avail_action = env.get_avail_actions()
        action = agent.choose_action(state, avail_action, evaluate=True)

        state, reward, done = env.step(action)

        ep_reward = ep_reward + reward
    return env.get_metric(), ep_reward

if __name__ == "__main__":
    start_time = time.time()
    for name in ["local", "edge", "cloud", "random", "greedy", "mlp", "gcn", "tom"]:
        episodes = 100
        dvr_rate_mean = 0
        ep_reward_mean = 0
        for i in range(episodes):
            dvr_rate, ep_reward = main(name, int(start_time)+i)
            dvr_rate_mean += dvr_rate / episodes
            ep_reward_mean += ep_reward / episodes
        print("agent: {}\t, dvr_rate_mean: {}\t\t, ep_reward_mean: {}".format(name, dvr_rate_mean, ep_reward_mean))