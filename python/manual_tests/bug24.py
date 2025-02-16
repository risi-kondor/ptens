import ptens
import ptens_base
import torch
import faulthandler

faulthandler.enable()
device = 'cuda:0'
# device = 'cpu'
# hidden_dim = 3
hidden_dim = 4
num_vertices = 10


G = ptens.ggraph.random(5, 0.5)
M = torch.randn(5, 3).to(device)
print("M is:", M)
f0 = ptens.subgraphlayer0.from_matrix(G, ptens.subgraph.trivial(), M)
print("f0 is:", f0)

S = ptens.subgraph.edge()
f1 = ptens.subgraphlayer1.gather(S, f0)
print("f1 is:", f1)

S = ptens.subgraph.trivial()
G1 = ptens.ggraph.random(num_vertices, 0.5)
G = G1

M = torch.randn(num_vertices, hidden_dim).to(device) 
print("M is:", M)
C = ptens.subgraphlayer0.from_matrix(G, S,  M)
print("C is:", C)

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
print("result:", result)

# --- bug ----
# gather0 = ptens.batched_subgraphlayer0.gather(edge, result)
gather0 = ptens.subgraphlayer0.gather(edge, result)
print("first gather succeeded:")
print("gather0.shape:", gather0.shape)
gather1 = ptens.subgraphlayer1.gather(edge, result)
print("second gather succeeded:")
print("gather1.shape:", gather1.shape)


atom = gather1.atoms
print("atom is:", atom)
catoms = ptens_base.catomspack.random(atom, 3)
print("catoms is:", catoms)
print("working so far")

print("compressing. calling csubgraphlayer1.compress")
compressed_gather = ptens.csubgraphlayer1.compress(catoms, gather1)
print("compressed_gather is:", compressed_gather)
