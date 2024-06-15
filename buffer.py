import numpy as np
from environment import Environment

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, env):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.env = env
        self.n_state = env.get_state_size()
        self.n_action = env.get_action_size()

        self.state = np.zeros((buffer_size, self.n_state))
        self.action = np.zeros((buffer_size, 1))
        self.reward = np.zeros((buffer_size, 1))
        self.next_state = np.zeros((buffer_size, self.n_state))
        self.avail_action = np.zeros((buffer_size, self.n_action))

        self.next_idx = 0
        self.num_in_buffer = 0
        pass

    def can_sample(self):
        return self.batch_size < self.num_in_buffer
        # return self.num_in_buffer >= self.buffer_size

    def sample(self):
        if self.can_sample():
            sample_index = np.random.choice(self.num_in_buffer, self.batch_size)
            batch_state = self.state[sample_index]
            batch_action = self.action[sample_index]
            batch_reward = self.reward[sample_index]
            batch_next_state = self.next_state[sample_index]
            batch_avail_action = self.avail_action[sample_index]

            return self.env.decode_batch_state(batch_state), batch_action, batch_reward, self.env.decode_batch_state(batch_next_state), batch_avail_action
        return None

    def store(self, state, action, reward, next_state, avail_action):

        self.state[self.next_idx] = self.env.encode_state(state)
        self.action[self.next_idx] = action
        self.reward[self.next_idx] = reward
        self.next_state[self.next_idx] = self.env.encode_state(next_state)
        self.avail_action[self.next_idx] = avail_action

        self.next_idx = (self.next_idx + 1) % self.buffer_size
        self.num_in_buffer = self.num_in_buffer + 1 if self.num_in_buffer < self.buffer_size else self.buffer_size