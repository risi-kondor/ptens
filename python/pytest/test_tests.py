import torch
import ptens
import pytest
import ptens_base
from conftest import numerical_grad_sum

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



graph_atoms_list= [
    (ptens.ggraph.from_edge_index( torch.Tensor([[0, 1], [1, 0]]).int() ),
     ptens_base.atomspack.from_list([[1]]),
     4),
]

class TestGather(object):
    h=1e-3

    def backprop(self,cls, G, atoms, nc, device):
        print(atoms)
        x=cls.sequential(atoms,nc).to(device)
        x.requires_grad_()
        print("x", x)
        atoms2 = G.subgraphs(ptens.subgraph.trivial())

        check = gradcheck(cls.gather, (atoms2, x), eps=self.h)
        assert check

        z = cls.gather(atoms2, x)
        print("z", z)
        z2 = cls.gather(atoms2, x*2)
        print("z2", z2)

        loss=torch.sum(z)
        print("loss", loss.item())
        loss.backward()
        xgrad=x.grad
        print("xgrad", xgrad)

        fn = lambda x: cls.gather(atoms2, x)
        xgrad2 = numerical_grad_sum(fn, x, self.h)
        print("xgrad2", xgrad2)
        assert torch.allclose(xgrad, xgrad2, rtol=1e-2, atol=1e-2)


        
    @pytest.mark.parametrize(('G', 'atoms', 'nc'), graph_atoms_list)
    def test_gather0(self,G, atoms, nc, device):
        N = max(G.adjacency_matrix().size())
        for i in range(len(atoms)):
            a = atoms[i]
            if len(a) > 0:
                assert max(a) < N
                
        self.backprop(ptens.ptensorlayer0, G, atoms, nc, device)

    # @pytest.mark.parametrize(('G', 'atoms', 'nc'), graph_atoms_list)
    # def test_gather1(self, G, atoms, nc, device):
    #     self.backprop(ptens.ptensorlayer0, G, atoms, nc, device)
