import torch
import ptens as p
import ptens_base 
import pytest


class TestGather(object):

    def backprop(self,src,fn,N,_nc):
        if(src==p.ptensor0):
            x=src.randn(N,_nc)
        else:
            atoms=ptens_base.atomspack.random(N,0.3)
            x=src.randn(atoms,_nc)
        x.requires_grad_()
        G=p.ggraph.random(N,0.3)
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
        self.backprop(p.ptensor0,p.gather,8,nc)
