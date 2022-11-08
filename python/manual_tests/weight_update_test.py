import torch
import ptens as p
N = 100
num_features = 60
num_classes = 20
x = p.ptensors1.randn([[i] for i in range(N)],num_features)
y = torch.randint(num_classes,size=(N,))

def test_weight_update(loss=torch.nn.NLLLoss(), optim=torch.optim.Adam):
  w = torch.nn.parameter.Parameter(torch.rand(num_features,num_classes))
  b = torch.nn.parameter.Parameter(torch.rand(num_classes))

  loss = torch.nn.NLLLoss()
  optim = optim([w,b],0.9)

  optim.zero_grad()
  loss(p.linmaps0(p.linear(x,w,b)).torch(),y).backward()
  optim.step()
  p.linear(x,w,b)
print("testing with NLLLoss")
test_weight_update()
print("test complete")
y = torch.nn.functional.one_hot(y,num_classes).float()
print("testing with MSELoss")
test_weight_update(torch.nn.MSELoss())
print("test complete")

