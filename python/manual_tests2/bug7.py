import torch
import ptens

n_nodes = 4
n_subnodes = 2
n_labels = 1
labels = torch.ones([n_nodes, n_labels])
for i in range(n_nodes):
    labels[i] *= i
sub_labels = labels[:n_subnodes]

n_edges = 6
last_one = n_nodes - 1
edge_index = []
sub_edge_index = []
for i in range(n_edges):
    other = int(i*11 + last_one) % n_nodes
    edge_index.append([last_one, other])
    if other < n_subnodes and last_one < n_subnodes:
        sub_edge_index.append([last_one, other])
    last_one = other

edge_index = torch.tensor(edge_index).T
sub_edge_index = torch.tensor(sub_edge_index).T


graph = ptens.ggraph.from_edge_index(edge_index.int(), labels=labels.int())
print(graph)

sub_graph = ptens.subgraph.from_edge_index(sub_edge_index.int(), n_subnodes)
print(sub_graph)

atoms = graph.subgraphs(sub_graph)

print(sub_graph, atoms)

sub_graph_with_label = ptens.subgraph.from_edge_index(sub_edge_index.int(), n_subnodes, labels=sub_labels.int())
print(sub_graph_with_label)

atoms = graph.subgraphs(sub_graph_with_label)
print(atoms)
print("expected: ", ([0,1],))




