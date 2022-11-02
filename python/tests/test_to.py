import torch
import ptens as p
import pytest

class TestTo(object):
    
    def backprop(self, pts, _atoms, _nc):
        x = pts.randn(_atoms, _nc)
        x.requires_grad_()
        z= x.to('cpu')

        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad() 

        xeps = x.randn_like()
        x_ = x+xeps
        z = x.to('cpu')
        xloss = z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[2,5],[1,2,6]]])
    @pytest.mark.parametrize('pts', [p.ptensors0, p.ptensors1, p.ptensors2])
    def test_to(self, pts, atoms, nc):
        self.backprop(pts, atoms, nc)
