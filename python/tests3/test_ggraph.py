import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" ggraph")
print("-------------------------------------\n")

print("A ggraph object represents a graph.\n")

print("The random constructor creates an Erdos-Renyi graph:\n")
G=p.ggraph.random(6,0.5)
print(G)

print("We can also define a graph from its adjaceny matrix:\n")
A=torch.tensor([[0,1,1],[1,0,1],[1,1,0]],dtype=torch.float)
G=p.ggraph.from_matrix(A)
print(G)

print("Or from its edge list:\n")
E=torch.tensor([[0,2,2,3],[2,0,3,2]],dtype=torch.float)
G=p.ggraph.from_edge_index(E)
print(G)

print("The adjacency matrix can be extracted as a PyTorch tensor:\n")
B=G.adjacency_matrix()
print(B)
print("\n")


print("\n---------")
print(" Labels")
print("---------\n")

print("We can create a labeled graph:\n")
L=torch.randn([A.size(0),5])
G2=p.ggraph.from_matrix(A,L)
print(G2)

print("Or add vertex labels to an existing graph:\n")
G3=p.ggraph.from_matrix(A)
G3.set_labels(L)
print(G3)

print("The label matrix can be extracted as a PyTorch tensor:\n")
print(G3.labels())
print("\n")


print("\n---------")
print(" Caching")
print("---------\n")


print("Graphs can be cached and later retrieved from the cache:\n")
G.cache(5)
H=p.ggraph.from_cache(5)
print(H)


print("\n-----------")
print(" Subgraphs")
print("-----------\n")

print("Given a subgraph S, we can find all occurrences of S in G:\n")
G=p.ggraph.random(8,0.5)
S=p.subgraph.triangle()
A=G.subgraphs(S)
print(A)
print("\n")

print("The result of finding subgraphs is automatically cached.")
print("We can inspect the graph's subgraph list cache:\n")
C=G.cached_subgraph_lists()
for c in C:
    print(c)
    print(C[c])
    print("\n")


print("\n-----------")
print(" Preloader")
print("-----------\n")

print("The preloader can enumerate all objects that were constructed from G:\n")

P=pb.ggraph_preloader(G.obj)
print(P)
print("\n")


