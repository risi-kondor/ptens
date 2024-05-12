import torch
import ptens as p
import ptens_base as pb 

G=p.ggraph.random(8,0.5)
print(G)

S=p.subgraph.triangle()
A=G.subgraphs(S)
print(A)

mlist=A.overlaps(A,1)
print(mlist.str(""))
print("\n")

pilot=pb.pilot0(A)
print(pilot)

mmap=pilot.mmap(mlist,pilot)
print(mmap)
