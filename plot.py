import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = eval(f.read())
    return np.array(data)

def plot(data):
    window_sizes = [10, 50, 100]
    plt.figure(figsize=(10, 6))
    for window_size in window_sizes:
        smoothed_data = moving_average(data, window_size)
        plt.plot(np.arange(window_size-1, len(data)), smoothed_data, label=f'Window size = {window_size}')
    plt.title('Episode Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file_path = os.path.join('saved', 'off', 'gat', '1736146194.1024332', 'ep_reward.txt')
    data = load_data(file_path)
    plot(data)
    