import torch
import ptens as p

# We can create a random graph: 

G=p.ggraph.random(6,0.5)
print(G)

# Or we can define it from a matrix:

A=torch.tensor([[0,1,1],[1,0,1],[1,1,0]],dtype=torch.float)
G=p.ggraph.from_matrix(A)
print(G)

# Or we can define it from its edge list:

E=torch.tensor([[0,2,2,3],[2,0,3,2]],dtype=torch.float)
G=p.ggraph.from_edge_index(E)
print(G)



