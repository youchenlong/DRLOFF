import numpy as np
from sko.GA import GA
from environment import Environment

env = Environment()

def main(seed=0):
    def obj_function(p):
        action_values = np.array(p)
        ep_reward, dvr_rate = run_an_episode(action_values, seed)
        return -ep_reward

    n_dim = len(env.G.nodes) * (env.M+2)
    ga = GA(func=obj_function, n_dim=n_dim, size_pop=20, max_iter=200, lb=[0 for _ in range(n_dim)], ub=[1 for _ in range(n_dim)])
    best_x, best_y = ga.run()
    return best_x, best_y

def run_an_episode(action_values, seed=0):
    ep_reward = 0
    env.reset(seed)
    action_values = action_values.reshape(len(env.G.nodes), env.M + 2)
    for i in range(len(env.G.nodes)):
        action_value = action_values[i]
        avail_action = env.get_avail_actions()
        action_value[avail_action == 0] = 0
        action = np.argmax(action_value)
        _, reward, _ = env.step(action)
        ep_reward += reward
    dvr_rate = env.get_metric()
    return ep_reward, dvr_rate

if __name__ == "__main__":
    ep_reward_mean = 0
    dvr_rate_mean = 0
    episodes = 10
    for i in range(episodes):
        best_x, best_y = main(i)
        ep_reward, dvr_rate = run_an_episode(best_x, i)
        assert ep_reward == -best_y, "error!"
        print("ep_reward: {}, dvr_rate:{}".format(ep_reward, dvr_rate))
        ep_reward_mean += ep_reward / episodes
        dvr_rate_mean += dvr_rate / episodes
    print(ep_reward_mean)
    print(dvr_rate_mean)

