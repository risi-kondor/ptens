import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" subgraphlayer2")
print("-------------------------------------\n")

print("A subgraphlayer2 is a second order P-tensor layer corresponding\nto the subgraphs induced by S in G.\n\n")

print("We can define a random second order subgraph layer:\n")
G=p.ggraph.random(8,0.5)
S=p.subgraph.triangle()
A=p.subgraphlayer2.randn(G,S,3)
print(A)

print("Or define it from a torch tensor:\n")
M=torch.randn([G.subgraphs(S).nrows2(),3])
A=p.subgraphlayer2.from_matrix(G,S,M)
print(A)

print("If two subgraphlayers have the same reference domains,\nwe can do arithmetic on them:\n")
B=p.subgraphlayer2.randn(G,S,3)
print(A+A)


print("\n---------")
print(" Linmaps")
print("---------\n")

print("Linmaps from a 0th order subgraph layer:\n")
A0=p.subgraphlayer0.randn(G,S,3)
B=p.subgraphlayer0.linmaps(A0)
print(B)

print("Linmaps from a 1st order subgraph layer:\n")
A1=p.subgraphlayer1.randn(G,S,3)
B=p.subgraphlayer2.linmaps(A1)
print(B)

print("Linmaps from a 2nd order subgraph layer:\n")
A2=p.subgraphlayer2.randn(G,S,3)
B=p.subgraphlayer2.linmaps(A2)
print(B)


print("\n---------")
print(" Gather")
print("---------\n")


atoms2=pb.atomspack.random(5,5,0.6)
print(atoms2,"\n")

B0=p.ptensorlayer2.gather(atoms2,A0)
print(B0)

B1=p.ptensorlayer2.gather(atoms2,A1)
print(B1)

B2=p.ptensorlayer2.gather(atoms2,A2)
print(B2)

