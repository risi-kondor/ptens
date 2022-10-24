import torch
import ptens as p
import ptens_base 
import pytest


class TestUnite(object):

    def backprop(self,src,fn,N,_nc):
        if(src==p.ptensors0):
            x=src.randn(N,_nc)
        else:
            atoms=ptens_base.atomspack.random(N,0.3)
            x=src.randn(atoms,_nc)
        x.requires_grad_()
        G=p.graph.random(N,0.3)
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
    def test_unite01(self,nc):
        self.backprop(p.ptensors0,p.unite1,8,nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_unite02(self,nc):
        self.backprop(p.ptensors0,p.unite2,8,nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_unite11(self,nc):
        self.backprop(p.ptensors1,p.unite1,8,nc)

    @pytest.mark.parametrize('nc', [1,2,4])
    def test_unite12(self,nc):
        self.backprop(p.ptensors1,p.unite2,8,nc)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    def test_unite21(self,nc):
        self.backprop(p.ptensors2,p.unite1,8,nc)

    @pytest.mark.parametrize('nc', [1,2,4])
    def test_unite22(self,nc):
        self.backprop(p.ptensors2,p.unite2,8,nc)


