import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import networkx as nx
import random
import time


class Constant():
    c_max = 5e10 # CPU频率最大值
    r_max = 8e6 # 传输速率最大值
    e_max = 5e10 # 存储容量最大值


def normalize(data):
    assert type(data) == np.ndarray, "data must be numpy ndarray"
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class Environment():

    def __init__(self):
        self.num_nodes = 100
        self.num_edges = 250
        # self.M = 3 # 基站数量
        self.M = 5 # 基站数量
        
        # 下面这行代码用于初始化成员变量，不可删除
        self.reset()
        pass

    def generate_dag(self, num_nodes, num_edges):
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        
        # Attempt to add edges without creating cycles
        while len(G.edges()) < num_edges:
            a, b = np.random.randint(0, num_nodes, size=2)
            if a != b and not G.has_edge(a, b):
                G.add_edge(a, b)
                # Check if adding this edge creates a cycle
                if nx.is_directed_acyclic_graph(G):
                    continue
                else:
                    G.remove_edge(a, b)
        
        # Convert to adjacency matrix
        adjacency_matrix = nx.to_numpy_array(G, dtype=int)
        return G, adjacency_matrix

    def generate_tolerance(self, V):
        # 均匀分布
        # t_v = np.zeros(V)
        # for i in range(V):
        #     if i == 0:
        #         t_v[i] = np.random.uniform(1, 3)
        #     else:
        #         t_v[i] = t_v[i-1] +  np.random.uniform(1, 3)

        # 正太分布（大于0）
        lower, upper = 0, 4
        mu, sigma = 2, 1
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        t_v = np.zeros(V)
        for i in range(V):
            if i == 0:
                t_v[i] = X.rvs()
            else:
                t_v[i] = t_v[i-1] + X.rvs()

        return t_v

    def generate_task(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        G, adjacency_matrix = self.generate_dag(self.num_nodes, self.num_edges)

        self.G = G
        self.adjacency_matrix = adjacency_matrix
        self.sorted_nodes = list(nx.topological_sort(G))
        V = len(self.sorted_nodes)
        self.current_idx = 0

        # 初始化任务的属性
        tasks = list(G.nodes)
        b_v = np.random.uniform(8e5, 1.6e6, V)  # 任务数据大小
        d_v = np.random.uniform(2e8, 2e9, V)  # 任务所需的CPU周期数

        # 重新排序任务
        sorted_index = self.sorted_nodes  # 获取按任务编号排序后的索引    
        # print(sorted_index)
        self.sorted_tasks = np.array(tasks)[sorted_index]  # 排序后的任务
        self.sorted_b_v = b_v[sorted_index]  # 按新任务顺序排列的任务数据大小
        self.sorted_d_v = d_v[sorted_index]  # 按新任务顺序排列的CPU周期数
        self.sorted_t_v = self.generate_tolerance(V) # 按新任务顺序排列的容忍延迟

        self.T_com = np.zeros(V) # 按新任务顺序排列的任务完成时间 
        self.completed = [False] * V # 任务是否完成
        self.free = [False] * V # 任务是否释放资源
        self.off_dev = [-1] * V # 任务执行设备
        pass


    def init_cluster(self):
        # TODO: 数据设计不合理

        # 本地
        self.l_vm = np.random.uniform(1e8, 2e8)  # 本地处理任务的CPU频率
        self.l_cap = np.random.uniform(1e8, 2e8) # 本地的存储容量
        # 边缘
        # self.e_vm = np.random.uniform(1e9, 2e9, self.M)  # 基站处理任务的CPU频率
        # self.r_vm = np.random.uniform(2e6, 4e6, self.M)  # 基站与任务之间的传输速率
        # self.e_cap = np.random.uniform(1e9, 2e9, self.M) # 基站的存储容量
        self.e_vm = np.zeros(self.M) # 基站处理任务的CPU频率
        self.r_vm = np.zeros(self.M) # 基站与任务之间的传输速率
        self.e_cap = np.zeros(self.M) # 基站的存储容量
        delta_e_vm = (2e9 - 1e9) / self.M
        delta_r_vm = (4e6 - 2e6) / self.M
        delta_e_cap = (2e9 - 1e9) / self.M
        for i in range(self.M):
            self.e_vm[i] = np.random.uniform(1e9 + i*delta_e_vm, 1e9 + (i+1)*delta_e_vm)
            self.r_vm[i] = np.random.uniform(1e6 + i*delta_r_vm, 1e6 + (i+1)*delta_r_vm)
            self.e_cap[i] = np.random.uniform(1e9 + i*delta_e_cap, 1e9 + (i+1)*delta_e_cap)
        # 云
        self.r_vc = np.random.uniform(2.4e6, 4.8e6)  # 云与任务之间的传输速率
        self.t_rt = 3  # 固定传输时延
        pass

    def reset(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)

        self.generate_task(self.num_nodes, self.num_edges)
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
                self.T_com[task_idx] = self.sorted_d_v[task_idx] / self.l_vm
            else:
                # 前一个节点
                last_T = self.T_com[task_idx-1]
                # 前继节点
                # task_ID = self.sorted_nodes[task_idx]
                # if list(self.G.predecessors(task_ID)) == []:
                #     pred_T = 0
                # else:
                #     pred_T = 0
                #     for ID in list(self.G.predecessors(task_ID)):
                #         pred_T = max(pred_T, self.T_com[np.where(self.sorted_nodes == ID)])
                pred_T = max(self.T_com[:task_idx])
                self.T_com[task_idx] = max(self.sorted_d_v[task_idx] / self.l_vm + pred_T, last_T)
            self.l_cap = self.l_cap - self.sorted_b_v[task_idx]
            self.off_dev[task_idx] = 0
        # 云
        elif action == self.M + 1:
            if task_idx == 0:
                self.T_com[task_idx] = self.sorted_b_v[task_idx] / self.r_vc + self.t_rt
            else:
                # 前一个节点
                last_T = self.T_com[task_idx-1]
                # 前继节点
                # task_ID = self.sorted_nodes[task_idx]
                # if list(self.G.predecessors(task_ID)) == []:
                #     pred_T = 0
                # else:
                #     pred_T = 0
                #     for ID in list(self.G.predecessors(task_ID)):
                #         pred_T = max(pred_T, self.T_com[np.where(self.sorted_nodes == ID)])
                pred_T = max(self.T_com[:task_idx])
                self.T_com[task_idx] = max(self.sorted_b_v[task_idx] / self.r_vc + self.t_rt + pred_T, last_T)
            self.off_dev[task_idx] = self.M + 1
        # 边缘
        else:
            edge_idx = action
            if task_idx == 0:
                self.T_com[task_idx] = self.sorted_d_v[task_idx] / self.e_vm[edge_idx] + self.sorted_b_v[task_idx] / self.r_vm[edge_idx]
            else:
                # 前一个节点
                last_T = self.T_com[task_idx-1]
                # 前继节点
                # task_ID = self.sorted_nodes[task_idx]
                # if list(self.G.predecessors(task_ID)) == []:
                #     pred_T = 0
                # else:
                #     pred_T = 0
                #     for ID in list(self.G.predecessors(task_ID)):
                #         pred_T = max(pred_T, self.T_com[np.where(self.sorted_nodes == ID)])
                pred_T = max(self.T_com[:task_idx])
                self.T_com[task_idx] = max(self.sorted_d_v[task_idx] / self.e_vm[edge_idx] + self.sorted_b_v[task_idx] / self.r_vm[edge_idx] + pred_T, last_T)
            self.e_cap[edge_idx] = self.e_cap[edge_idx] - self.sorted_b_v[task_idx]
            self.off_dev[task_idx] = edge_idx
        
        # 更新任务状态
        for idx in range(task_idx + 1):
            if self.T_com[idx] > 0:
                self.completed[idx] = True
        
        # 释放资源
        for idx in range(task_idx+1):
            if self.completed[idx] and not self.free[idx]:
                if self.off_dev[idx] == 0:
                    self.l_cap = self.l_cap + self.sorted_b_v[idx]
                elif self.off_dev[idx] == self.M + 1:
                    pass
                else:
                    edge_idx = self.off_dev[idx]
                    self.e_cap[edge_idx] = self.e_cap[edge_idx] + self.sorted_b_v[idx]
                self.free[idx] = True
        
        self.current_idx = self.current_idx + 1
        if self.current_idx == len(self.G.nodes):
            self.done = True
        return self.get_state(), self.get_reward(), self.done


    def get_state(self):
        if self.current_idx == len(self.G.nodes):
            task_idx = np.zeros(len(self.G.nodes))
        else:
            task_idx = np.zeros(len(self.G.nodes))
            task_idx[self.current_idx] = 1.0
        task_info = np.stack([normalize(self.sorted_b_v), normalize(self.sorted_d_v), normalize(self.sorted_t_v)], axis=1)

        dev_c = np.append(np.append(self.l_vm, self.e_vm), Constant.c_max)
        dev_r = np.append(np.append(Constant.r_max, self.r_vm), self.r_vc)
        dev_cap = np.append(np.append(self.l_cap, self.e_cap), Constant.e_max)
        dev_info = np.stack([normalize(dev_c), normalize(dev_r), normalize(dev_cap)], axis=1)
        
        state = (task_idx, task_info, dev_info, self.adjacency_matrix)
        return state

    def get_reward(self):
        # TODO： reward设计不合理，没法用于强化学习训练
        # DVR
        task_idx = self.current_idx - 1
        dvr = self.sorted_t_v[task_idx] - self.T_com[task_idx]
        r1 = dvr
        return r1

    def get_state_size(self):
        """
        task_idx + task_info + dev_info + graph
        """
        return len(self.G.nodes) + len(self.G.nodes) * 3 + (self.M + 2) * 3 + len(self.G.nodes) * len(self.G.nodes)

    def get_action_size(self):
        """
        1 + M + 1
        """
        return self.M + 2

    def encode_state(self, state):
        task_idx, task_info, dev_info, graph = state
        return np.hstack((task_idx.flatten(), task_info.flatten(), dev_info.flatten(), graph.flatten()))

    def encode_batch_state(self, batch_state):
        task_idx, task_info, dev_info, graph = batch_state
        batch_size = task_idx.shape[0]
        return np.hstack((task_idx.reshape(batch_size, -1), task_info.reshape((batch_size, -1)), dev_info.reshape((batch_size, -1)), graph.reshape((batch_size, -1))))

    def decode_batch_state(self, batch_state):
        num_nodes = len(self.G.nodes)
        M = self.M
        
        task_idx_dim = num_nodes
        task_info_dim = num_nodes * 3
        dev_info_dim = (M + 2) * 3
        graph_dim = num_nodes * num_nodes

        task_idx = np.array([item.reshape(num_nodes, 1) for item in batch_state[:, :task_idx_dim]])
        task_info = np.array([item.reshape(num_nodes, 3) for item in batch_state[:, task_idx_dim:task_idx_dim+task_info_dim]])
        dev_info = np.array([item.reshape((M+2), 3) for item in batch_state[:, task_idx_dim+task_info_dim:task_idx_dim+task_info_dim+dev_info_dim]])
        graph = np.array([item.reshape(num_nodes, num_nodes) for item in batch_state[:, -graph_dim:]])
        return (task_idx, task_info, dev_info, graph)

    def get_avail_actions(self):
        avail_actions = np.ones(self.M + 2)
        task_idx = self.current_idx
        if self.sorted_b_v[task_idx] > self.l_cap:
            avail_actions[0] = 0
        for edge_idx in range(1, self.M):
            if self.sorted_b_v[task_idx] > self.e_cap[edge_idx]:
                avail_actions[edge_idx] = 0
        return avail_actions

    def log(self):
        # 任务
        print("排序后的任务顺序:", self.sorted_tasks)
        print("排序后的任务数据大小:", self.sorted_b_v)
        print("排序后的CPU周期数:", self.sorted_d_v)
        print("排序后的容忍时间:", self.sorted_t_v)
        print("任务完成时间:", self.T_com)
        # 设备
        print("基站与任务之间的传输速率:", self.r_vm)
        print("基站处理任务的CPU频率:", self.e_vm)
        print("基站的存储容量:", self.e_cap)
        print("云与任务之间的传输速率:", self.r_vc)
        print("固定传输时延:", self.t_rt)
        # 本地
        print("本地处理任务的CPU频率:", self.l_vm)
        print("本地的存储容量:", self.l_cap)

    def plot_task(self):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.G)  # Positions for all nodes
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, font_size=16, arrows=True)
        plt.title("Directed Acyclic Graph (DAG)")
        plt.show()

    def get_metric(self):
        # DVR
        task_idx = self.current_idx
        dvr_count = 0
        for idx in range(task_idx):
            if self.T_com[idx] > self.sorted_t_v[idx]:
                dvr_count = dvr_count + 1
        return dvr_count / len(self.G.nodes)


def main():
    env = Environment()
    env.reset()
    env.log()

if __name__ == "__main__":
    main()