import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import networkx as nx
import random
import time

class Constant:
    c_max = 5e10 # CPU频率最大值
    r_max = 8e6 # 传输速率最大值
    s_max = 5e10 # 存储容量最大值


def normalize(data):
    assert type(data) == np.ndarray, "data must be numpy ndarray"
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class Environment:

    def __init__(self):
        self.ID = 0 # 环境唯一标识
        self.adjs = {} # 所有任务的邻接矩阵

        # self.min_num_nodes = 80 # 最小节点数
        # self.max_num_nodes = 100 # 最大节点数
        # self.min_num_edges = 200 # 最小边数
        # self.max_num_edges = 250 # 最大边数
        self.min_num_nodes = 100
        self.max_num_nodes = 100
        self.min_num_edges = 250
        self.max_num_edges = 250
        self.M = 5 # 基站数量
        pass

    def generate_dag(self, num_nodes, num_edges):
        # Create a directed acyclic graph
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        while G.number_of_edges() < num_edges:
            a, b = np.random.randint(0, num_nodes, size=2)
            if a != b and not G.has_edge(a, b):
                G.add_edge(a, b)
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(a, b)
        
        return G

    def generate_tolerance(self, G, queue):
        # 正态分布
        lower, upper = 1, 6
        mu, sigma = 4, 1
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        # plot
        # plt.hist(X.rvs(1000), bins=100, density=True)
        tolerance = np.zeros(len(queue))
        for task_idx in queue:
            tolerance[task_idx] = X.rvs()
            dependencies = list(G.predecessors(task_idx))
            if len(dependencies) > 0:
                tolerance[task_idx] = tolerance[task_idx] + max([tolerance[dep] for dep in dependencies])
        sorted_tolerance = np.array([tolerance[task_idx] for task_idx in queue])
        return sorted_tolerance


    def generate_task(self):
        # 任务拓扑图
        self.G = self.generate_dag(self.num_nodes, self.num_edges)
        self.adjs[self.ID] = np.pad(nx.to_numpy_array(self.G, dtype=int), \
                                ((0, self.max_num_nodes - self.num_nodes), (0, self.max_num_nodes - self.num_nodes)))
        
        # 调度队列
        self.queue = np.array(list(nx.topological_sort(self.G)))
        
        # 任务属性
        self.data_size = np.random.uniform(8e5, 1.6e6, self.num_nodes)
        self.cpu_cycles = np.random.uniform(2e8, 2e9, self.num_nodes)
        self.tolerance = self.generate_tolerance(self.G, self.queue)

        # 任务状态信息
        self.current_idx = 0
        self.completed = [False] * self.num_nodes # 任务是否完成
        self.T_com = np.zeros(self.num_nodes) # 任务完成时间 
        self.free = [False] * self.num_nodes # 任务是否释放资源
        self.off_dev = [-1] * self.num_nodes # 任务执行设备
        pass


    def init_cluster(self):
        # TODO: 数据设计不合理

        # 本地
        self.local_cpu_cycles = np.random.uniform(1e8, 2e8)  # 本地处理任务的CPU频率
        self.local_storage = np.random.uniform(1e8, 2e8) # 本地的存储容量
        # 边缘
        self.edge_cpu_cycles = np.zeros(self.M) # 基站的CPU频率
        self.edge_trans_rate = np.zeros(self.M) # 基站的传输速率
        self.edge_storage = np.zeros(self.M) # 基站的存储容量
        delta_edge_cpu_cycles = (2e9 - 1e9) / self.M
        delta_edge_trans_rate = (4e6 - 2e6) / self.M
        delta_edge_storage = (2e9 - 1e9) / self.M
        for i in range(self.M):
            self.edge_cpu_cycles[i] = np.random.uniform(1e9 + i*delta_edge_cpu_cycles, 1e9 + (i+1)*delta_edge_cpu_cycles)
            self.edge_trans_rate[i] = np.random.uniform(1e6 + i*delta_edge_trans_rate, 1e6 + (i+1)*delta_edge_trans_rate)
            self.edge_storage[i] = np.random.uniform(1e9 + i*delta_edge_storage, 1e9 + (i+1)*delta_edge_storage)
        # 云
        self.cloud_trans_rate = np.random.uniform(2.4e6, 4.8e6)  # 云的传输速率
        self.cloud_fixed_time = 3  # 固定传输时延
        pass

    def reset(self, seed=0):
        # 简化训练复杂度：例如节点数可选值有5个，边可选值有5个
        random.seed(seed)
        np.random.seed(seed)

        self.ID = self.ID + 1
        self.num_nodes = random.choice(range(self.min_num_nodes, self.max_num_nodes + 5, 5))
        self.num_edges = random.choice(range(self.min_num_edges, self.max_num_edges + 10, 10))

        self.generate_task()
        self.init_cluster()
        self.done = False
        
        return self.get_state()

    def step(self, action):
        """
        action: [0, 1, 2, ... M, M+1]
        edge: 0, 1, 2, ..., M-1
        local: M
        cloud: M+1
        """
        task_idx = self.current_idx

        # 本地
        if action == self.M:
            if task_idx == 0:
                self.T_com[task_idx] = self.cpu_cycles[task_idx] / self.local_cpu_cycles
            else:
                # 调度队列中前一个节点
                last_T = self.T_com[task_idx-1]
                # 图中前继节点
                pred_T = max(self.T_com[:task_idx])
                self.T_com[task_idx] = max(self.cpu_cycles[task_idx] / self.local_cpu_cycles + pred_T, last_T)
            self.local_storage = self.local_storage - self.data_size[task_idx]
            self.off_dev[task_idx] = 0
        # 云
        elif action == self.M + 1:
            if task_idx == 0:
                self.T_com[task_idx] = self.data_size[task_idx] / self.cloud_trans_rate + self.cloud_fixed_time
            else:
                last_T = self.T_com[task_idx-1]
                pred_T = max(self.T_com[:task_idx])
                self.T_com[task_idx] = max(self.data_size[task_idx] / self.cloud_trans_rate + self.cloud_fixed_time + pred_T, last_T)
            self.off_dev[task_idx] = self.M + 1
        # 边缘
        else:
            edge_idx = action
            if task_idx == 0:
                self.T_com[task_idx] = self.cpu_cycles[task_idx] / self.edge_cpu_cycles[edge_idx] + self.data_size[task_idx] / self.edge_trans_rate[edge_idx]
            else:
                last_T = self.T_com[task_idx-1]
                pred_T = max(self.T_com[:task_idx])
                self.T_com[task_idx] = max(self.cpu_cycles[task_idx] / self.edge_cpu_cycles[edge_idx] + self.data_size[task_idx] / self.edge_trans_rate[edge_idx] + pred_T, last_T)
            self.edge_storage[edge_idx] = self.edge_storage[edge_idx] - self.data_size[task_idx]
            self.off_dev[task_idx] = edge_idx
        
        # 更新任务状态
        for idx in range(task_idx + 1):
            if self.T_com[idx] > 0:
                self.completed[idx] = True
        for idx in range(task_idx + 1):
            if self.completed[idx] and not self.free[idx]:
                if self.off_dev[idx] == 0:
                    self.local_storage = self.local_storage + self.data_size[idx]
                elif self.off_dev[idx] == self.M + 1:
                    pass
                else:
                    edge_idx = self.off_dev[idx]
                    self.edge_storage[edge_idx] = self.edge_storage[edge_idx] + self.data_size[idx]
                self.free[idx] = True
        
        self.current_idx = self.current_idx + 1
        if self.current_idx == len(self.G.nodes):
            self.done = True
        return self.get_state(), self.get_reward(), self.done


    def get_state(self):
        task_idx = np.zeros(self.max_num_nodes)
        if self.current_idx < self.num_nodes:
            task_idx[self.current_idx] = 1.0
        task_info_padding = np.stack([np.pad(normalize(self.data_size), (0, self.max_num_nodes - self.num_nodes)), \
                                        np.pad(normalize(self.cpu_cycles), (0, self.max_num_nodes - self.num_nodes)), \
                                        np.pad(normalize(self.tolerance), (0, self.max_num_nodes - self.num_nodes))], axis=1)

        dev_cpu_cycles = np.append(np.append(self.local_cpu_cycles, self.edge_cpu_cycles), Constant.c_max)
        dev_trans_rate = np.append(np.append(Constant.r_max, self.edge_trans_rate), self.cloud_trans_rate)
        dev_storage = np.append(np.append(self.local_storage, self.edge_storage), Constant.s_max)
        dev_info = np.stack([normalize(dev_cpu_cycles), normalize(dev_trans_rate), normalize(dev_storage)], axis=1)
        
        state = (task_idx, task_info_padding, dev_info)
        return state

    def get_reward(self):
        # TODO： reward设计不合理，没法用于强化学习训练
        # DVR
        # 注意这里是self.current_idx - 1而不是self.current_idx
        task_idx = self.current_idx - 1
        r1 = (self.tolerance[task_idx] - self.T_com[task_idx]) / sum(self.tolerance)
        return r1

    def get_state_size(self):
        """
        task_idx + task_info + dev_info
        """
        return self.max_num_nodes + self.max_num_nodes * 3 + (self.M + 2) * 3

    def get_action_size(self):
        return self.M + 2

    def encode_state(self, state):
        task_idx, task_info, dev_info = state
        return np.hstack((task_idx.flatten(), task_info.flatten(), dev_info.flatten()))

    def encode_batch_state(self, batch_state):
        task_idx, task_info, dev_info = batch_state
        batch_size = task_idx.shape[0]
        return np.hstack((task_idx.reshape(batch_size, -1), task_info.reshape((batch_size, -1)), dev_info.reshape((batch_size, -1))))

    def decode_batch_state(self, batch_state):
        max_num_nodes = self.max_num_nodes
        M = self.M
        
        task_idx_dim = max_num_nodes
        task_info_dim = max_num_nodes * 3
        dev_info_dim = (M + 2) * 3

        task_idx = np.array([item.reshape(max_num_nodes, 1) for item in batch_state[:, :task_idx_dim]])
        task_info = np.array([item.reshape(max_num_nodes, 3) for item in batch_state[:, task_idx_dim:task_idx_dim+task_info_dim]])
        dev_info = np.array([item.reshape((M+2), 3) for item in batch_state[:, task_idx_dim+task_info_dim:task_idx_dim+task_info_dim+dev_info_dim]])
        return (task_idx, task_info, dev_info)

    def get_avail_actions(self):
        avail_actions = np.ones(self.M + 2)
        task_idx = self.current_idx
        if self.data_size[task_idx] > self.local_storage:
            avail_actions[0] = 0
        for edge_idx in range(1, self.M):
            if self.data_size[task_idx] > self.edge_storage[edge_idx]:
                avail_actions[edge_idx] = 0
        return avail_actions

    def get_metric(self):
        task_idx = self.current_idx
        dvr_count = 0
        for idx in range(task_idx):
            if self.T_com[idx] > self.tolerance[idx]:
                dvr_count = dvr_count + 1
        return dvr_count / len(self.G.nodes)

    def update_adjs(self, IDs):
        self.adjs = {key: value for key, value in self.adjs.items() if key in IDs}

    def log(self):
        # 任务
        print("调度顺序:", self.queue)
        print("任务数据大小:", self.data_size)
        print("任务CPU周期数:", self.cpu_cycles)
        print("任务容忍时间:", self.tolerance)
        # 设备
        print("本地的CPU频率:", self.local_cpu_cycles)
        print("本地的存储容量:", self.local_storage)
        print("基站的CPU频率:", self.edge_cpu_cycles)
        print("基站的传输速率:", self.edge_trans_rate)
        print("基站的存储容量:", self.edge_storage)
        print("云的传输速率:", self.cloud_trans_rate)
        print("固定传输时延:", self.cloud_fixed_time)   

    def plot_task(self):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.G)  # Positions for all nodes
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, font_size=16, arrows=True)
        plt.title("Directed Acyclic Graph (DAG)")
        plt.show()


def main():
    env = Environment()
    env.reset()
    env.log()
    env.plot_task()

if __name__ == "__main__":
    main()