import torch
from torch_geometric.nn import Sequential
import ptens as p
#
learning_rate = 0.001
decay = 0.9
channels_in = 128
channels_out = 1
model = Sequential('x,G',[
  (p.modules.Linear(channels_in,128),'x->x'),
  (lambda x,G: p.unite1(x,G),'x,G->x'),
  p.linmaps0,
  lambda x: x.torch(),
  torch.nn.Linear(128,channels_out),
])
model.cuda()

optim = torch.optim.Adam(model.parameters(),learning_rate)
loss = torch.nn.MSELoss()

N = 50
G = p.graph.random(N,0.1)
x = p.ptensors0.randn([[i] for i in range(N)],channels_in).to('cuda')
optim.zero_grad()
y = torch.rand(N).to('cuda')
l = loss(model(x,G).flatten(),y)
l.backward()
optim.step()
