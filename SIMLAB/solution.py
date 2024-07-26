import json
from dataclasses import dataclass
import numpy as np


@dataclass
class Node:
    i: int
    p: float
    adj: list[tuple[int, int]]


# Load configuration
with open("config.json", 'r') as f:
    config = json.load(f)
max_iter = config['max_iter']
eps = config['eps']
P_in = config['P_in']
P_out = config['P_out']
n = config['n']
m = config['m']
k_h = config['k_h']
k_v = config['k_v']

# Build a network
nodes = dict()
nodes_cnt = 0
for i in range(1, n + 1):
    for j in range(1, m + 1):
        if (i, j) == (1, 1):
            index = -1
            p = P_in
        elif (i, j) == (n, m):
            index = -1
            p = P_out
        else:
            index = nodes_cnt
            nodes_cnt += 1
            p = (P_in + P_out) / 2
        nodes[(i, j)] = Node(index, p, [])
edges = dict()
for i, j in nodes:
    if j < m:
        edges[((i, j), (i, j + 1))] = edges[((i, j + 1), (i, j))] = k_h[i - 1][j - 1]
        nodes[(i, j)].adj.append((i, j + 1))
        nodes[(i, j + 1)].adj.append((i, j))
    if i < n:
        edges[((i, j), (i + 1, j))] = edges[((i + 1, j), (i, j))] = k_v[i - 1][j - 1]
        nodes[(i, j)].adj.append((i + 1, j))
        nodes[(i + 1, j)].adj.append((i, j))

# Update pressures iteratively using Newton's method
for iter_cnt in range(max_iter):
    derivative_matrix = np.zeros((nodes_cnt, nodes_cnt))
    F_vector = np.zeros(nodes_cnt)
    for (i, j), node in nodes.items():
        if node.i != -1:
            F = 0.
            dF_dp_self = 0.
            for i_adj, j_adj in node.adj:
                adj_node = nodes[(i_adj, j_adj)]
                k = edges[((i, j), (i_adj, j_adj))]
                if adj_node.p >= node.p:  # inflow
                    F += np.sqrt((adj_node.p - node.p) / k)
                    dF_dp = 1 / (2 * np.sqrt(k * (adj_node.p - node.p + eps * node.p)))
                else:  # outflow
                    F -= np.sqrt((node.p - adj_node.p) / k)
                    dF_dp = 1 / (2 * np.sqrt(k * (node.p - adj_node.p + eps * node.p)))
                dF_dp_self -= dF_dp
                if adj_node.i != -1:
                    derivative_matrix[node.i][adj_node.i] = dF_dp
            derivative_matrix[node.i][node.i] = dF_dp_self
            F_vector[node.i] = -F
    d_p_vector = np.linalg.solve(derivative_matrix, F_vector)
    for node in nodes.values():
        if node.i != -1:
            node.p += d_p_vector[node.i]
    if abs(max(d_p_vector)) / ((P_in + P_out) / 2) < eps:
        print(f"Solution converged after {iter_cnt + 1} iterations\n")
        break
else:
    print(f"Max number of iterations ({max_iter}) reached\n")

# Visualize the result
sign = lambda x: 1 if x >= 0 else -1
for i in range(1, n + 1):
    for j in range(1, m + 1):
        print(nodes[(i, j)].p, end="," if j < m else "\n")
        if j < m:
            p_diff = nodes[(i, j)].p - nodes[(i, j + 1)].p
            Q_h = sign(p_diff) * np.sqrt(abs(p_diff) / edges[((i, j), (i, j + 1))])
            print(Q_h, end=",")
    if i < n:
        for j in range(1, m + 1):
            p_diff = nodes[(i, j)].p - nodes[(i + 1, j)].p
            Q_v = sign(p_diff) * np.sqrt(abs(p_diff) / edges[((i, j), (i + 1, j))])
            print(Q_v, end=",," if j < m else "\n")
