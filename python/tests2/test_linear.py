import torch
import ptens as p
import pytest

class TestLinear(object):
    
    def backprop(self, pts, fn, _w, _b, _atoms, _nc):
        x = pts.randn(_atoms, _nc)
        x.requires_grad_()
        z=fn(x,_w,_b)

        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad() 

        xeps = pts.randn(_atoms, _nc)
        z = fn(x+xeps,_w,_b)
        xloss = z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('w', [torch.tensor([[1.0,0.0],[0.0,1.0],[0.0,0.0]])])
    @pytest.mark.parametrize('b', [torch.tensor([1.0,2.0,3.0])])
    def test_linear0(self, w, b):
        self.backprop(p.ptensors0b, p.linear, torch.randn([3,3]), b, [[1,2,3],[3,5],[2]], 3)

    @pytest.mark.parametrize('w', [torch.tensor([[1.0,0.0],[0.0,1.0],[0.0,0.0]])])
    @pytest.mark.parametrize('b', [torch.tensor([1.0,2.0,3.0])])
    def test_linear1(self, w, b):
        self.backprop(p.ptensors1b, p.linear, torch.randn([3,3]), b, [[1,2,3],[3,5],[2]], 3)

    @pytest.mark.parametrize('w', [torch.tensor([[1.0,0.0],[0.0,1.0],[0.0,0.0]])])
    @pytest.mark.parametrize('b', [torch.tensor([1.0,2.0,3.0])])
    def test_linear2(self, w, b):
        self.backprop(p.ptensors2b, p.linear, torch.randn([3,3]), b, [[1,2,3],[3,5],[2]], 3)

