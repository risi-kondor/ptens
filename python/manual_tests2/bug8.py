import numpy as np
import torch
from torch_geometric.data import Data
import ptens

import networkx as nx

device="cpu"

def get_nx_graph(n=6):
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(i, label=i+1)
        graph.add_edge(i, (i+1)%n, label=i)

    graph.add_edge(0, n//2, label=n+1)
    return graph

def get_nx_subgraphs():
    ret_list = []
    a = nx.Graph()
    a.add_node(0, label=1)
    a.add_node(1, label=2)
    a.add_node(2, label=3)
    a.add_edge(0, 1, label=0)
    a.add_edge(1, 2, label=1)
    ret_list += [a]
    return ret_list

def nx_to_pyg(nx_graph):
    senders = []
    receivers = []
    edge_attr = []
    for edge in nx_graph.edges(data=True):
        senders += [edge[0]]
        receivers += [edge[1]]
        edge_attr += [(edge[2]["label"])]

    senders = np.asarray(senders, dtype=int)
    receivers = np.asarray(receivers, dtype=int)
    edge_attr = np.asarray(edge_attr, dtype=int)

    edge_index = torch.from_numpy(np.stack([senders, receivers]))
    edge_attr = torch.from_numpy(edge_attr)

    x = np.zeros((nx_graph.number_of_nodes(), 1))
    for node in nx_graph.nodes(data=True):
        x[node[0]] = [node[1]["label"]]
    x = torch.from_numpy(x)

    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index)


def nx_to_ggraph(nx_graph):
    pyg_graph = nx_to_pyg(nx_graph)
    n = pyg_graph.x.shape[0]
    edge_index = pyg_graph.edge_index.int()
    ggraph = ptens.ggraph.from_edge_index(edge_index, n, labels=pyg_graph.x.int())
    return ggraph

def nx_to_subgraph(nx_graph):
    pyg_graph = nx_to_pyg(nx_graph)
    n = pyg_graph.x.shape[0]
    edge_index = pyg_graph.edge_index.int()
    subgraph = ptens.subgraph.from_edge_index(edge_index, n, labels=pyg_graph.x.int())
    return subgraph


nx_graph = get_nx_graph()
nx_subgraphs = get_nx_subgraphs()

graph = nx_to_ggraph(nx_graph)
subgraphs = [nx_to_subgraph(sg) for sg in nx_subgraphs]

M = graph.labels().T

subgraph_layer = ptens.subgraphlayer0.from_matrix(G=graph, S=subgraphs[0], M=M)
print(subgraph_layer)
print(subgraphs[0])
a=graph.subgraphs(subgraphs[0])
print(a)

f1 = ptens.subgraphlayer1.gather(subgraphs[0], subgraph_layer)
print(f1)
