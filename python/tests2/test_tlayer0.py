import torch
import ptens as p
import pytest

class TestTlayer0(object):

    def backprop(self,fn,_atoms,_nc):
        x=p.tlayer.randn(_atoms,_nc)
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
        self.backprop(p.linmaps0,[[1],[2],[3]],nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_linmaps1(self,nc):
        self.backprop(p.linmaps1,[[1],[2],[3]],nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_linmaps2(self,nc):
        self.backprop(p.linmaps2,[[1],[2],[3]],nc)

    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    def test_linmaps0(self,atoms):
        self.backprop(p.linmaps0,atoms, 2)

    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,3],[2,7,6]]])
    def test_linmaps1(self,atoms):
        self.backprop(p.linmaps1,atoms, 4)

    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2],[6,8,9]]])
    def test_linmaps2(self,atoms):
        self.backprop(p.linmaps2,atoms, 5)
