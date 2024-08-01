import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" ggraph_cache")
print("-------------------------------------\n")

print("ggraph_cache is a global object that can be used to stash graphs.\n")

G1=p.ggraph.random(6,0.5)
G1.cache(3)
print(G1)

G2=p.ggraph.random(6,0.5)
G2.cache(4)
print(G2)

#H=pb.ggraph_cache.graph(3)
H=p.ggraph.from_cache(3) 
print(H)

print("Number of cached graphs: ",pb.ggraph_cache.size(),"\n")

print(pb.ggraph_cache.str())


