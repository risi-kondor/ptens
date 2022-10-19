import torch
import ptens as p
import ptens_base
import pytest

class TestOuter(object):
    
    def backprop(self, pts_1, pts_2, fn, _atoms, _nc):
        x = pts_1.randn(_atoms, _nc)
        y = pts_2.randn(_atoms, _nc)
        x.requires_grad_()
        y.requires_grad_()
        z=fn(x,y)

        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad() 

        xeps = x.randn_like(0.01)
        yeps = y.randn_like(0.01)
        z = fn(x+xeps,y+yeps)
        xloss = z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    @pytest.mark.parametrize('pts_1', [p.ptensors0, p.ptensors1, p.ptensors2])
    @pytest.mark.parametrize('pts_2', [p.ptensors0, p.ptensors1, p.ptensors2])
    def test_outer0(self, pts_1, pts_2, nc, atoms):
        self.backprop(pts_1, pts_2, p.cat, atoms, nc)


