import torch
import ptens
from ptens_base import atomspack
from torch.autograd.gradcheck import gradcheck, gradgradcheck

def test_bug1(device):
    nnodes = 15
    graph =ptens.ggraph.random(nnodes,0.5)
    print(graph)
    subgraphs = [ptens.subgraph.trivial(), ptens.subgraph.edge()]
    node_values = torch.rand(nnodes, 1, requires_grad=True)

    node_attributes = ptens.subgraphlayer0.from_matrix(graph, ptens.subgraph.trivial(), node_values)


    for sg in subgraphs:
        gather_features = ptens.subgraphlayer0.gather(sg, node_attributes)
        result = torch.sum(gather_features)
        result.backward()
        print(node_values.grad)

        # linmap_features = ptens.subgraphlayer0.linmaps(node_attributes)
        result = torch.sum(node_attributes)
        result.backward()
        print(node_attributes.grad)

        check = gradcheck(ptens.subgraphlayer0.gather, (sg, node_attributes), eps=1e-3)
        print(check)

