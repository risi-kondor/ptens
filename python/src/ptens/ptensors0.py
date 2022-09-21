import torch

from ptens_base import ptensors0 as _ptensors0


class ptensors0(torch.Tensor):

    @classmethod
    def raw(self, _atoms, _nc, _dev=0):
        R=ptensors0(1)
        R.obj=_ptensors0.raw(_atoms,_nc,_dev)
        return R

    @classmethod
    def zeros(self, _atoms, _nc, _dev=0):
        R=ptensors0(1)
        R.obj=_ptensors0.zero(_atoms,_nc,_dev)
        return R

    @classmethod
    def randn(self, _atoms, _nc, _dev=0):
        R=ptensors0(1)
        R.obj=_ptensors0.gaussian(_atoms,_nc,_dev)
        return R

    @classmethod
    def sequential(self, _atoms, _nc, _dev=0):
        R=ptensors0(1)
        R.obj=_ptensors0.sequential(_atoms,_nc,_dev)
        return R


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=ptensors0(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=ptensors0(1)
        R.obj=self.obj.view_of_grad()
        return R


    def get_nc(self):
        return self.obj.get_nc()

    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def push_back(self, x):
        return self.obj.push_back(x)



    
    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


