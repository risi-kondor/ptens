import torch
import ptens as p

c = 2
y=torch.randn(c,1) 
print("y= ", y, y.shape)

x0=p.ptensors0.randn([[1,2],[3]],c)
print("x0= ", x0, x0.shape)
out0 = x0.mult_channels(y)
print("out0= ", out0)

x1=p.ptensors1.randn([[1,2],[3]],c)
print("x1= ", x1, x1.shape) 
out1 = x1.mult_channels(y)
print("out1= ", out1)

x2=p.ptensors2.randn([[1,2],[3]],c)  
print("x2= ", x2, x2.shape)
out2 = x2.mult_channels(y)
print("out2= ", out2)



