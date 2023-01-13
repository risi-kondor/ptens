import torch
import ptens
epochs = 50
# must be 64
in_channels = 64
num_classes = 7
N=70
w = torch.rand(in_channels*2,num_classes,device='cuda')
b = torch.rand(num_classes,device='cuda')
# we put in a single edge somewhere.
#adj = torch.tensor([7,8,8,7],dtype=int).unsqueeze(1)
#adj = torch.sparse_coo_tensor(adj, torch.ones(adj.size(1)),size=(N,N)).to_dense()
# we add a 64 degree node, connected to the node of degree > 0 from before.
# Only crashes for 3 < pos
adj=torch.zeros(N,N)
pos = 68
adj[pos,:] = 1
adj[:,pos] = 1
adj[pos,pos] = 0
adj[7,8]=1
adj[8,7]=1
G = ptens.graph.from_matrix(adj)
print(G)
x = ptens.ptensors1.randn(G.nhoods(1),in_channels).to('cuda')
#print(G.nhoods(1))
epochs=50
for epoch in range(epochs):
  print(epoch)
  print("\ta")
  # It crashes during the execution of 'unite1.'
  a = ptens.unite1(x,G)
  print("\tb")
  ptens.linear(a,w,b)
  print("\tc")
  
