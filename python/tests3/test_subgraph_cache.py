import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" subgraph_cache")
print("-------------------------------------\n")

print("Every subgraph created in ptens is automatically cached.\n")

print(p.subgraph.edge())
print(p.subgraph.triangle())
print(p.subgraph.cycle(5))
print(p.subgraph.star(5))

print(pb.subgraph_cache.str())


