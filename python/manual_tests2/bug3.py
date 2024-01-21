import numpy as np
import ptens
import faulthandler
faulthandler.enable()

with open("fault_handler.log", "w") as fobj:
    G = ptens.ggraph.random(10, 0.5)
    cycle11 = ptens.subgraph.cycle(11)
    print(cycle11)
    print(cycle11.torch())
    print(cycle11.torch().shape)
    # x = ptens.ptensors1b.randn([ for i in range(10)], 2)
    x = ptens.ptensors1b.sequential([[i] for i in range(10)], 128)
    # print("x is:", x)
    x.requires_grad = True
    pred = ptens.subgraphlayer1b.gather_from_ptensors(x, G, cycle11)
    print("pred shape:", pred.torch().shape)
    print("pred is:", pred)
    pred.torch().sigmoid().sum().backward()
    
