import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" subgraphlayer1")
print("-------------------------------------\n")

print("A subgraphlayer1 is a first order P-tensor layer corresponding\nto the subgraphs induced by S in G.\n\n")

print("We can define a random first order subgraph layer:\n")
G=p.ggraph.random(8,0.5)
S=p.subgraph.triangle()
A=p.subgraphlayer1.randn(G,S,3)
print(A)

print("Or define it from a torch tensor:\n")
M=torch.randn([G.subgraphs(S).nrows1(),3])
A=p.subgraphlayer1.from_matrix(G,S,M)
print(A)
z=torch.relu(A)
print(z)
print(z.atoms)

print("If two subgraphlayers have the same reference domains,\nwe can do arithmetic on them:\n")
B=p.subgraphlayer1.randn(G,S,3)
print(A+A)


print("\n---------")
print(" Linmaps")
print("---------\n")

print("Linmaps from a 0th order subgraph layer:\n")
A=p.subgraphlayer0.randn(G,S,3)
B=p.subgraphlayer0.linmaps(A)
print(B)

print("Linmaps from a 1st order subgraph layer:\n")
A=p.subgraphlayer1.randn(G,S,3)
B=p.subgraphlayer1.linmaps(A)
print(B)

print("Linmaps from a 2nd order subgraph layer:\n")
A=p.subgraphlayer2.randn(G,S,3)
B=p.subgraphlayer1.linmaps(A)
print(B)


print("\n---------")
print(" Gather")
print("---------\n")

E=p.subgraph.edge()


print("Gather from a 0th order subgraph layer:\n")
A0=p.subgraphlayer0.randn(G,E,3)
B0=p.subgraphlayer1.gather(S,A0)
print(B0)

print("Gather from a 1st order subgraph layer:\n")
A1=p.subgraphlayer1.randn(G,E,3)
B1=p.subgraphlayer1.gather(S,A1)
print(B1)


print("Gather from a 2nd order subgraph layer:\n")
A2=p.subgraphlayer2.randn(G,E,3)
B2=p.subgraphlayer1.gather(S,A2)
print(B2)

P=pb.ggraph_preloader(G.obj)
print(P)
print("\n")


