import torch
import ptens as p
import ptens_base as pb


print("\n----------------------------------")
print(" subgraph")
print("----------------------------------\n")

ix=torch.tensor([[0,1,2],[1,2,0]],dtype=torch.int)
d=torch.tensor([4,3,3],dtype=torch.int)

print("Subgraphs have their own class.\n")
print("There are a number of predefined classes of subgraphs:\n")

print(p.subgraph.edge())
print(p.subgraph.triangle())
print(p.subgraph.cycle(5))
print(p.subgraph.star(5))

print("--- Triangle ---")
s0=p.subgraph.from_edge_index(ix,3)
print(s0)

print("--- Triangle with degrees ---")
s1=p.subgraph.from_edge_index(ix,degrees=d)
print(s1)
#print(G.subgraphs(s1))

print("Contents of cache:\n")
C=pb.subgraph_cache.torch()
for s in C:
    print(s)
