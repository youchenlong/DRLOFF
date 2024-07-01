### Management
* `agent.py`
    - [x] DQN
* `baselines.py`
    - [x] all local
    - [x] all edge
    - [x] all cloud
    - [x] random
    - [x] greedy
* `buffer.py`
* `environment.py`
    * Task
        - data size: $[8\times10^5, 1.6\times10^6]$
        - cpu cycles: $[2\times10^8, 2\times10^9]$
    * Local
        - cpu: $[1\times10^8, 2\times10^8]$
        - storage: $[1\times10^8, 2\times10^8]$
    * Edge
        - cpu: $[1\times10^9, 2\times10^9]$
        - trans: $[2\times10^6, 4\times10^6]$
        - storage: $[1\times10^9, 2\times10^9]$
    * Cloud
        - trans: $[2.4\times10^6, 4.8\times10^6]$
        - fixed trans time: $3$
* `gnn.py`
    - [x] GCN
    - [ ] GAT
* `scheduler.py`

### MDP
* state
    - state space: [task_idx, task_info, dev_info, adj]
* action
    - action space: M + 1 + 1
    - 0~M-1: edge
    - M: local
    - M+1: cloud
* reward
    - DVR (deadline violation ratio)

### train and run
* train DQN
    ```python
    python agent.py
    ```
* run all algs
    ```python
    python main.py
    ```