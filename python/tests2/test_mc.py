import torch
import ptens as p
import pytest
class TestMc(object):
    
    def backprop(self,k,_atoms,_nc):
        x = p.tlayer.randn(k, _atoms, _nc)
        y = torch.randn(_nc) 
        x.requires_grad_()
        z=x.mult_channels(y)

        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps = x.randn_like()
        x_ = x+xeps
        z = x_.mult_channels(y)
        xloss = z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('order', [0, 1, 2])
    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    def test_mc(self, order, nc, atoms):
        self.backprop(order, atoms, nc)

