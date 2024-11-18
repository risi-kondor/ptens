import ptens
import torch

# device = 'cuda:0'
device = 'cpu'
hidden_dim = 256
num_vertices = 10
atom = torch.arange(num_vertices).reshape(-1,1).tolist()
A=ptens.ptensorlayer0.randn(atom,hidden_dim).to(device)
# S = ptens.subgraph.edge()
S = ptens.subgraph.trivial()
G1 = ptens.ggraph.random(num_vertices, 0.5)
G = ptens.batched_ggraph.from_graphs([G1])
C = ptens.batched_subgraphlayer0.from_ptensorlayers([A,A,A])
# C = ptens.subgraphlayer0.from_ptensorlayers(G, S, [A])
# print("C is:", C)


edge = ptens.subgraph.edge()

linear_weight = torch.randn(hidden_dim, hidden_dim, device=device)

# linear_layer = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
# batch_layer = torch.nn.BatchNorm1d(hidden_dim).to(device)
relu_layer = torch.nn.ReLU().to(device)
#result = relu_layer(linear_layer(C))
# result = torch.matmul(C, linear_weight)
result = C
# result = batch_layer(C)
# result = relu_layer(C)
# print("result stride:", result.stride())
# print("result shape:", result.shape)
print("result is:", result)

# --- bug ----
gather0 = ptens.batched_subgraphlayer0.gather(edge, result)
# # gather0 = ptens.subgraphlayer0.gather(edge, result)
print("first gather succeeded")
gather1 = ptens.batched_subgraphlayer1.gather(edge, result)
print("both gathers succeeded")
