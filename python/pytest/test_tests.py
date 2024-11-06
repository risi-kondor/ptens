import torch
import ptens
import pytest
import ptens_base

from torch.autograd.gradcheck import gradcheck


def test_bug1(device):
    nnodes = 15
    graph = ptens.ggraph.random(nnodes, 0.5)
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



class TestGather(object):

    def backprop(self,cls,fn,N,_nc):
        if(cls==ptens.ptensor0):
            x=cls.randn(N,_nc)
        else:
            atoms=ptens_base.atomspack.random(N,0.3)
            x=cls.randn(atoms,_nc)
        x.requires_grad_()
        G=ptens.ggraph.random(N,0.3)
        z=fn(x,G)
        
        testvec=z.randn_like()
        loss=z.inp(testvec).to('cuda')
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps=x.randn_like()
        z=fn(x+xeps,G)
        xloss=z.inp(testvec).to('cuda')
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_gather(self,nc):
        self.backprop(ptens.ptensor0,ptens.gather,8,nc)
