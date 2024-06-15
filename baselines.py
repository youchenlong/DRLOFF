import numpy as np

class LocalAgent():
    def __init__(self):
        pass
    
    def choose_action(self, state, avail_action):
        action = len(avail_action) - 2 
        return action

class EdgeAgent():
    def __init__(self):
        pass
    
    def choose_action(self, state, avail_action):
        M = len(avail_action) - 2
        avail_action = avail_action[:M]
        action = np.random.choice(len(avail_action), p=avail_action/sum(avail_action))
        return action

class CloudAgent():
    def __init__(self):
        pass
    
    def choose_action(self, state, avail_action):
        action = len(avail_action) - 1
        return action

class RandomAgent():
    def __init__(self):
        pass
    
    def choose_action(self, state, avail_action):
        action = np.random.choice(len(avail_action), p=avail_action/sum(avail_action))
        return action

class GreedyAgent():
    def __init__(self):
        pass
    
    def choose_action(self, state, avail_action):
        M = len(avail_action) - 2
        task_idx, task_info, dev_info, graph = state
        w_comp = 0.8
        w_trans = 0.2
        v_comp = dev_info[:M, 0] / sum(dev_info[:M, 0])
        v_trans = dev_info[:M, 1] / sum(dev_info[:M, 1])
        action = np.random.choice(M, p=w_comp*v_comp+w_trans*v_trans)
        return action