import torch
import ptens as p
import pytest

class TestLinear(object):
    
    def backprop(self, k, fn, _w, _b, _atoms, _nc):
        x = p.tlayer.randn(k,_atoms, _nc)
        x.requires_grad_()
        z=fn(x,_w,_b)

        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad() 

        xeps = p.tlayer.randn(k,_atoms, _nc)
        z = fn(x+xeps,_w,_b)
        xloss = z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('k', [0,1,2])
    @pytest.mark.parametrize('w', [torch.tensor([[1.0,0.0,2.0],[0.0,1.0,4.0],[0.0,0.0,-1.0]])])
    @pytest.mark.parametrize('b', [torch.tensor([1.0,2.0,3.0])])
    def test_linear0(self, k, w, b):
        self.backprop(k, p.linear, w, b, [[1,2,3],[3,5],[2]], 3)


