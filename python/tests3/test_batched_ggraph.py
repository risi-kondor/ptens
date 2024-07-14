import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" batched_ggraph")
print("-------------------------------------\n")

print("A batched_ggraph holds a batch of ggraph objects.\n")

print("We can create it from a collection of individual ggraphs:\n")

G0=p.ggraph.random(6,0.5)
G1=p.ggraph.random(6,0.5)
G2=p.ggraph.random(6,0.5)

B=p.batched_ggraph.from_graphs([G0,G1,G2])
print(B)

print("\nOr from cached graphs:\n")
G0.cache(5)
C=p.batched_ggraph.from_cache([5,5,5])
print(C)

print("Given a subgraph S, we can find all occurrences of S in G:\n")
S=p.subgraph.triangle()
atoms=B.subgraphs(S)
print(atoms)
print("\n")

