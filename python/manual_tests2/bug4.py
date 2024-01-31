import torch
import ptens as p
import torch_geometric.utils as pyg_utils

# G = p.ggraph.random(6,0.6)
device = 'cuda'
G = p.subgraph.cycle(6)
G = p.ggraph.from_matrix(G.torch())
n = 6
edges = pyg_utils.dense_to_sparse(G.torch())[0]

tria = p.subgraph.triangle()
edge = p.subgraph.edge()
node = p.subgraph.trivial()
cycle5 = p.subgraph.cycle(5)
cycle6 = p.subgraph.cycle(6)
cycle7 = p.subgraph.cycle(7)


#M = torch.arange(n).view(n,1).float().to(device)
M = torch.randn([n,2]).to(device)
M.requires_grad = True
print(M.shape)
A = p.ptensors0b.from_matrix(M)
print("A\n",A)
F = p.subgraphlayer1b.gather_from_ptensors(A,G,cycle6)
print("F\n",F)
autobahn = p.Autobahn(2,2,cycle6).to(device)

print(autobahn.w)
      
F = autobahn(F)
print("F\n",F)
pred = F.torch().sum()
print(f"pred: {pred}")
pred.backward()
