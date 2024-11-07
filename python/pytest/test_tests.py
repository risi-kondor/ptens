import torch
import ptens
import pytest
import ptens_base

from torch.autograd.gradcheck import gradcheck


def test_bug1(device):
    nnodes = 15
    graph = ptens.ggraph.random(nnodes, 0.5)
    subgraphs = [ptens.subgraph.trivial(), ptens.subgraph.edge()]
    node_values = torch.rand(nnodes, 1, requires_grad=True)

    node_attributes = ptens.subgraphlayer0.from_matrix(graph, ptens.subgraph.trivial(), node_values)

    for sg in subgraphs:
        gather_features = ptens.subgraphlayer0.gather(sg, node_attributes)
        result = torch.sum(gather_features)
        result.backward()

        # linmap_features = ptens.subgraphlayer0.linmaps(node_attributes)
        result = torch.sum(node_attributes)
        result.backward()

        check = gradcheck(ptens.subgraphlayer0.gather, (sg, node_attributes), eps=1e-3)
        assert check



class TestGather(object):

    
    def backprop(self,cls, N,nc, device):
        atoms=ptens_base.atomspack.random(N, nc, 0.3)
        x=cls.randn(atoms,nc).to(device)
        x.requires_grad_()
        G=ptens.ggraph.random(N,0.3)
        atoms2 = G.subgraphs(ptens.subgraph.trivial())

        check = gradcheck(cls.gather, (atoms2, x), eps=1e-3)
        assert check

        z = cls.gather(atoms2, x)
        loss=torch.sum(z)
        loss.backward()
        xgrad=x.grad
        print("xgrad", xgrad)


        h=1e-6
        xgrad2 = torch.zero_like(xgrad)
        for i in range(xgrad2.size()):
            xp = copy.deepcopy(x)
            xm = copy.deepcopy(x)
            xp[i] += h
            xm[i] -= h
            
            grad[i] = cls(gather

        z_plus = cls.gather(atoms2, x+h)
        z_minus = cls.gather(atoms2, x-h)
        xgrad2 = (z_plus - z_minus)/(2*h)
        print("xgrad2", xgrad2)
        

        
    @pytest.mark.parametrize(('N', 'nc'), [(8, 1), (1, 2), (16, 4)])
    def test_gather0(self,N, nc, device):
        self.backprop(ptens.ptensorlayer0,N,nc, device)

    @pytest.mark.parametrize(('N', 'nc'), [(8, 1), (1, 2), (16, 4)])
    def test_gather1(self,N, nc, device):
        self.backprop(ptens.ptensorlayer0,N,nc, device)
