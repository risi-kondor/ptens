# %%
import torch
import ptens as p
import torch_geometric.utils as pyg_utils

# %%
import networkx as nx
import matplotlib.pyplot as plt


def vis_graph(edge_index):
    
    # Create a new graph
    edge_index = edge_index.transpose(1,0).tolist()
    print(edge_index)
    G = nx.Graph(edge_index)

    # Draw the graph
    nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
    plt.show()

def _scalar_mult(x, alpha):
    return x.mult_channels(alpha.broadcast_to(x.get_nc()).clone())

# %%
# G = p.ggraph.random(6,0.6)
device = 'cpu'
E =  torch.tensor([[0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 1],
        [0, 1, 1, 1, 1, 0]]).float()
G = p.ggraph.from_matrix(E)
edges = pyg_utils.dense_to_sparse(G.torch())[0]
vis_graph(edges)

tria = p.subgraph.triangle()
edge = p.subgraph.edge()
node = p.subgraph.trivial()
cycle5 = p.subgraph.cycle(5)
cycle6 = p.subgraph.cycle(6)
cycle7 = p.subgraph.cycle(7)



M = torch.arange(6).view(6,1).float()
A = p.ptensors0b.from_matrix(M)
print("A\n",A)
B = p.subgraphlayer1b.gather_from_ptensors(A,G,tria)
print("B\n",B)

F = p.subgraphlayer1b.gather_from_ptensors(B,G,cycle5)
print("F\n",F,F.get_nc())
print("F\n",F)

