import numpy as np

class Episode:
    def __init__(self, env):
        self.n_state = env.get_state_size()
        self.n_action = env.get_action_size()

    def update(self, states, actions, rewards, next_states, avail_actions, ID):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.avail_actions = avail_actions
        self.ID = ID

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.episode_in_buffer = 0
        self.buffer = []
        
    def can_sample(self):
        return self.batch_size < self.episode_in_buffer

    def sample(self):
        if self.can_sample():
            indices = np.random.choice(self.episode_in_buffer, self.batch_size, replace=False)
            episodes = [self.buffer[i] for i in indices]
            return episodes
        return None

    def insert_an_episode(self, episode):
        self.buffer.append(episode)
        self.episode_in_buffer = self.episode_in_buffer + 1 if self.episode_in_buffer < self.buffer_size else self.buffer_size

    def get_IDs(self):
        return [episode.ID for episode in self.buffer]