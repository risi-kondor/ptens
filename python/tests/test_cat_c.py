import torch
import ptens as p
import pytest

class TestCat(object):
    
    def backprop(self, pts, fn, _atoms, _nc):
        x = pts.randn(_atoms, _nc, 0.1)
        y = pts.randn(_atoms, _nc, 0.1)
        x.requires_grad_()
        y.requires_grad_()
        z=fn(x,y)

        testvec=z.randn_like(1.0)
        loss=z.inp(testvec).to('cuda')
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()
        ygrad=y.get_grad() 

        xeps = x.randn_like(1.0)
        yeps = y.randn_like(1.0)
        z = fn(x+xeps,y+yeps)
        xloss = z.inp(testvec).to('cuda')
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    def test_cat0(self, nc, atoms):
        self.backprop(p.ptensors0, p.cat, nc, atoms)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    def test_cat1(self, nc, atoms):
        self.backprop(p.ptensors1, p.cat, nc, atoms)

    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('atoms', [[[1],[2],[6]],[[1],[2,5],[1,2,6]]])
    def test_cat2(self, nc, atoms):
        self.backprop(p.ptensors2, p.cat, nc, atoms)

