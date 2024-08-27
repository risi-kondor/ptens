import torch
import ptens_base as pb
import ptens as p

print("\n-------------------------------------")
print (" csubgraphlayer2")
print("-------------------------------------\n")

print("A compressed subgraphlayer2 is a second order P-tensor layer corresponding\nto the subgraphs induced by S in G.\n\n")
nvecs=3

print("We can define a random second order subgraph layer:\n")
G=p.ggraph.random(8,0.5)
S=p.subgraph.triangle()
A=p.csubgraphlayer2.randn(G,S,nvecs,3)
print(A)

print("Or define it from a torch tensor:\n")
M=torch.randn([G.subgraphs(S).nrows1(),nvecs,3])
A=p.csubgraphlayer2.from_tensor(G,S,M)
print(A)

print("If two subgraphlayers have the same reference domains,\nwe can do arithmetic on them:\n")
B=p.csubgraphlayer2.randn(G,S,nvecs,3)
print(A+A)


print("\n---------")
print(" Linmaps")
print("---------\n")

#print("Linmaps from a 0th order subgraph layer:\n")
#A=p.csubgraphlayer0.randn(G,S,nvecs,3)
#B=p.csubgraphlayer0.linmaps(A)
#print(B)

print("Linmaps from a 1st order subgraph layer:\n")
A=p.csubgraphlayer2.randn(G,S,nvecs,3)
B=p.csubgraphlayer2.linmaps(A)
print(B)

print("Linmaps from a 2nd order subgraph layer:\n")
A=p.csubgraphlayer2.randn(G,S,nvecs,3)
B=p.csubgraphlayer2.linmaps(A)
print(B)

print("\n---------")
print(" Gather")
print("---------\n")

E=p.subgraph.edge()


#print("Gather from a 0th order subgraph layer:\n")
#A0=p.csubgraphlayer0.randn(G,E,nvecs,3)
#B0=p.csubgraphlayer2.gather(S,A0)
#print(B0)

#print("Gather from a 1st order subgraph layer:\n")
#A1=p.csubgraphlayer2.randn(G,S,3,3)
#B1=p.csubgraphlayer2.gather(S,A1)
#print(B1)


#print("Gather from a 2nd order subgraph layer:\n")
#A2=p.csubgraphlayer2.randn(G,E,nvecs,3)
#B2=p.csubgraphlayer2.gather(S,A2)
#print(B2)

P=pb.ggraph_preloader(G.obj)
print(P)
print("\n")


