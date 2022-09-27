import torch
import ptens as p
import pytest

class TestPtensors0(object):

    def backprop(self,fn,_nc):
        _atoms=[[1],[2],[3]]
        x=p.ptensors0.randn(_atoms,_nc)
        x.requires_grad_()
        z=fn(x)
        
        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps=p.ptensors0.randn(_atoms,_nc)
        z=fn(x+xeps)
        xloss=z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_linmaps0(self,nc):
        self.backprop(p.linmaps0,nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_linmaps1(self,nc):
        self.backprop(p.linmaps1,nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_linmaps2(self,nc):
        self.backprop(p.linmaps2,nc)

