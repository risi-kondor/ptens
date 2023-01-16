import torch
import ptens as p
import  pytest
class TestTransfer(object):
    def backprop(self,ptensorsk,_atoms,_nc, new_atoms):
        x=ptensorsk.randn(_atoms,_nc, _device=None)
        x.requires_grad_()
        z=p.transfer0(x,new_atoms)
        
        testvec=z.randn_like()
        loss=z.inp(testvec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()

        xeps=ptensorsk.randn(_atoms,_nc, _device=None)
        z=p.transfer0(x+xeps,new_atoms)
        xloss=z.inp(testvec)
        assert(torch.allclose(xloss-loss,xeps.inp(xgrad),rtol=1e-3, atol=1e-4))

   # @pytest.mark.parametrize('ptensorsk', [p.ptensors0, p.ptensors1, p.ptensors2])
   # @pytest.mark.parametrize('fn', [p.transfer0, p.transfer1, p.transfer2])
    @pytest.mark.parametrize('atoms', [[[1,2,3],[3,5],[2]]])
   # @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('new_atoms', [[[1,2],[2,5,1,2,6]]])
    def test_px_transferx(self,atoms,new_atoms):
        self.backprop(p.ptensors0,atoms,2,new_atoms)

        
