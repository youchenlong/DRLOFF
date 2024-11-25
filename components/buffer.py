import numpy as np
from collections import deque

from env.environment import Environment

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, env):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.env = env
        self.n_state = env.get_state_size()
        self.n_action = env.get_action_size()

        self.state = deque(maxlen=buffer_size)
        self.action = deque(maxlen=buffer_size)
        self.reward = deque(maxlen=buffer_size)
        self.next_state = deque(maxlen=buffer_size)
        self.avail_action = deque(maxlen=buffer_size)
        self.IDs = deque(maxlen=buffer_size)

    def can_sample(self):
        return self.batch_size <= len(self.state)

    def sample(self):
        if self.can_sample():
            indices = np.random.choice(len(self.state), self.batch_size, replace=False)
            batch_state = [self.state[i] for i in indices]
            batch_action = [self.action[i] for i in indices]
            batch_reward = [self.reward[i] for i in indices]
            batch_next_state = [self.next_state[i] for i in indices]
            batch_avail_action = [self.avail_action[i] for i in indices]
            batch_IDs = [self.IDs[i] for i in indices]

            return self.env.decode_batch_state(np.array(batch_state)), \
                    np.array(batch_action), \
                    np.array(batch_reward), \
                    self.env.decode_batch_state(np.array(batch_next_state)), \
                    np.array(batch_avail_action), \
                    np.array(batch_IDs)
        return None

    def store(self, state, action, reward, next_state, avail_action):
        self.state.append(self.env.encode_state(state))
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(self.env.encode_state(next_state))
        self.avail_action.append(avail_action)
        self.IDs.append(self.env.ID)