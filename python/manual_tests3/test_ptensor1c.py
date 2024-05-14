import torch
import ptens as p

a=p.ptensor1c.randn([1,3,4],3)
b=p.ptensor1c.randn([1,3,4],3)
print(a)
print(b)

c=a+b
print(c)

relu=torch.nn.ReLU(inplace=True)
d=relu(a)

w=torch.randn(3,5)
d=torch.mm(a,w)
print(d)
