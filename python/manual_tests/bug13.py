import ptens
import torch

device = 'cuda:0'
# device = 'cpu'
# hidden_dim = 3
hidden_dim = 256
num_vertices = 10
S = ptens.subgraph.trivial()
G1 = ptens.ggraph.random(num_vertices, 0.5)
G = G1

M = torch.randn(num_vertices, hidden_dim).to(device) 
C = ptens.subgraphlayer0.from_matrix(G, S,  M)

edge = ptens.subgraph.edge()

linear_layer = torch.nn.Linear(hidden_dim, hidden_dim).to(device)
# batch_layer = torch.nn.BatchNorm1d(hidden_dim).to(device)
# relu_layer = torch.nn.ReLU().to(device)
# print("linear layer's weight:", linear_layer.weight)

result = linear_layer(C)
# result = C
# result = batch_layer(C)
# result = relu_layer(C)
# print("result.atoms:", result.atoms)
# print("result repr:", result.__repr__())
# print("result:", result)

# --- bug ----
# gather0 = ptens.batched_subgraphlayer0.gather(edge, result)
gather0 = ptens.subgraphlayer0.gather(edge, result)
print("first gather succeeded:")
gather1 = ptens.subgraphlayer1.gather(edge, result)
print("second gather succeeded:")
