import torch
import ptens

n_nodes = 4
n_labels = 2
labels = torch.ones([n_nodes, n_labels])
for i in range(n_nodes):
    labels[i] *= i

n_edges = 6
last_one = n_nodes - 1
edge_index = []
for i in range(n_edges):
    other = int(i*11 + last_one) % n_nodes
    edge_index.append([last_one, other])
    last_one = other
edge_index = torch.tensor(edge_index).T
print(edge_index)

edge_index = edge_index.float()
# labels = labels.float()
graph = ptens.ggraph.from_edge_index(edge_index)
print(graph)

