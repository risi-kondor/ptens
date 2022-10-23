import torch
import ptens as p
import pytest

class TestPtensor0(object):

    def backprop(self,pt,fn,_atoms,_nc):
        x=pt.randn(_atoms,_nc)
        x.requires_grad_()
        z=fn(x)
        
        testvec=fn(pt.randn(_atoms,_nc))
        
        loss=torch.dot(torch.tensor(z), torch.tensor(testvec))
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps=pt.randn(_atoms,_nc)
        z=fn(x+xeps)
        xloss=torch.dot(torch.tensor(z), torch.tensor(testvec))
        assert(torch.allclose(xloss-loss,torch.dot(torch.tensor(xeps), torch.tensor(xgrad)),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_ptensor0_linmaps0(self, nc):
        self.backprop(p.ptensor0, p.linmaps0, [1,2,3], nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_ptensor0_linmaps1(self, nc):
        self.backprop(p.ptensor0, p.linmaps1, [1,2,3], nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_ptensor0_linmaps2(self, nc):
        self.backprop(p.ptensor0, p.linmaps2, [1,2,3], nc)


