import torch

import ptens_base
from ptens_base import ptensors2 as _ptensors2

import ptens.ptensors0 
import ptens.ptensors1 


class ptensors2(torch.Tensor):

    @classmethod
    def raw(self, _atoms, _nc, _dev=0):
        R=ptensors2(1)
        R.obj=_ptensors2.raw(_atoms,_nc,_dev)
        return R

    @classmethod
    def zeros(self, _atoms, _nc, _dev=0):
        R=ptensors2(1)
        R.obj=_ptensors2.zero(_atoms,_nc,_dev)
        return R

    @classmethod
    def randn(self, _atoms, _nc, _dev=0):
        R=ptensors2(1)
        R.obj=_ptensors2.gaussian(_atoms,_nc,_dev)
        return R

    @classmethod
    def sequential(self, _atoms, _nc, _dev=0):
        R=ptensors2(1)
        R.obj=_ptensors2.sequential(_atoms,_nc,_dev)
        return R


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=ptensors2(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=ptensors2(1)
        R.obj=self.obj.view_of_grad()
        return R


    def get_nc(self):
        return self.obj.get_nc()

    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def __getitem__(self,i):
        return Ptensors2_getFn.apply(self,i)
    
    def push_back(self, x):
        return self.obj.push_back(x)


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def __mul__(self,y):
        return Ptensors0_mprodFn.apply(self,y)
    

    def linmaps0(self):
        return Ptensors2_Linmaps0Fn.apply(self);

    def linmaps1(self):
        return Ptensors2_Linmaps1Fn.apply(self);

    def linmaps2(self):
        return Ptensors2_Linmaps2Fn.apply(self);


    def transfer0(self,_atoms):
        return Ptensors2_Transfer0Fn.apply(self,_atoms)

    def transfer1(self,_atoms):
        return Ptensors2_Transfer1Fn.apply(self,_atoms)

    def transfer2(self,_atoms):
        return Ptensors2_Transfer2Fn.apply(self,_atoms)

    
    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------


class Ptensors2_getFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,i):
        R=ptens.ptensor2(x.obj[i].torch())
        R.atoms=x.atoms_of(i)
        ctx.x=x.obj
        ctx.i=i
        return R

    @staticmethod
    def backward(ctx,g):
        R=ptensors2(1)
        ctx.x.add_to_grad(ctx.i,g)
        return R, None
    

class Ptensors2_mprodFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),y.size(1))
        R.obj.add_mprod(x.obj,y)
        ctx.x=x.obj
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,x,y):
        pass


class Ptensors2_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*2)
        ptens_base.add_linmaps2to0(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps2to0_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors2(1)


class Ptensors2_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*5)
        ptens_base.add_linmaps2to1(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps2to1_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors2(1)


class Ptensors2_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*15)
        ptens_base.add_linmaps2to2(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps2to2_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors2(1)


