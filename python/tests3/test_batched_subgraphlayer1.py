import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" batched_subgraphlayer1")
print("-------------------------------------\n")

print("A batched_subgraphlayer1 is a batch of subgraphlayer1 objects.\n")

print("Let's first define a batch of graphs and a subgraph\n")
G0=p.ggraph.random(6,0.5)
G1=p.ggraph.random(6,0.5)
G2=p.ggraph.random(6,0.5)
G=p.batched_ggraph.from_graphs([G0,G1,G2])
print(G)
S=p.subgraph.triangle()
print(S)

print("We can then define a random batched first order subgraph layer:\n")
A=p.batched_subgraphlayer1.randn(G,S,3)
print(A)

print("Or define it from a torch tensor:\n")
M=torch.randn([G.subgraphs(S).nrows1(),3])
A=p.batched_subgraphlayer1.from_matrix(G,S,M)
print(A)

print("If two batched subgraphlayers have the same reference domains,\nwe can do arithmetic on them:\n")
B=p.batched_subgraphlayer1.randn(G,S,3)
print(A+A)


print("\n---------")
print(" Linmaps")
print("---------\n")

print("Linmaps from a 0th order subgraph layer:\n")
A=p.batched_subgraphlayer1.from_matrix(G,S,M)
B=p.batched_subgraphlayer1.linmaps(A)
print(B)

print("Linmaps from a 1st order subgraph layer:\n")
A=p.batched_subgraphlayer1.randn(G,S,3)
B=p.batched_subgraphlayer1.linmaps(A)
print(B)

print("Linmaps from a 2nd order subgraph layer:\n")
A=p.batched_subgraphlayer2.randn(G,S,3)
B=p.batched_subgraphlayer1.linmaps(A)
print(B)


print("\n---------")
print(" Gather")
print("---------\n")

E=p.subgraph.edge()


print("Gather from a 0th order subgraph layer:\n")
A0=p.batched_subgraphlayer1.randn(G,E,3)
B0=p.batched_subgraphlayer1.gather(S,A0)
print(B0)

print("Gather from a 1st order subgraph layer:\n")
A1=p.batched_subgraphlayer1.randn(G,E,3)
B1=p.batched_subgraphlayer1.gather(S,A1)
print(B1)


print("Gather from a 2nd order subgraph layer:\n")
A2=p.batched_subgraphlayer2.randn(G,E,3)
B2=p.batched_subgraphlayer1.gather(S,A2)
print(B2)



