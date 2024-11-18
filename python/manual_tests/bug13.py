import ptens
import torch

device = 'cuda:0'
#device = 'cpu'
# hidden_dim = 3
hidden_dim = 2
num_vertices = 5
S = ptens.subgraph.trivial()
G1 = ptens.ggraph.random(num_vertices, 0.5)
G = G1

M = torch.randn(num_vertices, hidden_dim).to(device) 
C = ptens.subgraphlayer0.from_matrix(G, S,  M)
C=ptens.batched_subgraphlayer0.from_subgraphlayers([C])

edge = ptens.subgraph.edge()

linear_layer = torch.nn.Linear(hidden_dim, hidden_dim).to(device)
# batch_layer = torch.nn.BatchNorm1d(hidden_dim).to(device)
#relu_layer = torch.nn.ReLU().to(device)
# print("linear layer's weight:", linear_layer.weight)

#result = ptens.batched_subgraphlayer0(linear_layer(C))
result = C
print(type(result))
# result = batch_layer(C)
# result = relu_layer(C)
# print("result.atoms:", result.atoms)
# print("result repr:", result.__repr__())
# print("result:", result)

# --- bug ----
gather0 = ptens.batched_subgraphlayer0.gather(edge, result)
print("first gather succeeded:")
gather1 = ptens.batched_subgraphlayer1.gather(edge, result)
print("second gather succeeded:")
