import torch

import ptens_base 
from ptens_base import ptensors1 as _ptensors1

import ptens.ptensors0 
import ptens.ptensors2 


class ptensors1(torch.Tensor):

    @classmethod
    def raw(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.raw(_atoms,_nc,_dev)
        return R

    @classmethod
    def zeros(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.zero(_atoms,_nc,_dev)
        return R

    @classmethod
    def randn(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.gaussian(_atoms,_nc,_dev)
        return R

    @classmethod
    def sequential(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.sequential(_atoms,_nc,_dev)
        return R


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=ptensors1(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=ptensors1(1)
        R.obj=self.obj.view_of_grad()
        return R


    def get_nc(self):
        return self.obj.get_nc()

    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def push_back(self, x):
        return self.obj.push_back(x)


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def linmaps0(self):
        return Ptensors1_Linmaps0Fn.apply(self);

    def linmaps1(self):
        return Ptensors1_Linmaps1Fn.apply(self);

    def linmaps2(self):
        return Ptensors1_Linmaps2Fn.apply(self);


    def transfer0(self,_atoms):
        return Ptensors1_Transfer0Fn.apply(self,_atoms)

    def transfer1(self,_atoms):
        return Ptensors1_Transfer1Fn.apply(self,_atoms)

    def transfer2(self,_atoms):
        return Ptensors1_Transfer2Fn.apply(self,_atoms)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()

# ------------------------------------------------------------------------------------------------------------


class Ptensors1_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        R=ptens.ptensors0.zeros(x.obj.get_atomsref(),x.obj.get_nc())
        ptens_base.add_linmaps1to0(R.obj,x.obj)
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps1to0_back(ctx.x.obj.get_gradp(),g.obj)
        return ptensors1(1)







