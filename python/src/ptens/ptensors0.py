import torch

import ptens_base
from ptens_base import ptensors0 as _ptensors0

import ptens.ptensor0
import ptens.ptensors1
import ptens.ptensors2 


class ptensors0(torch.Tensor):

    @classmethod
    def from_matrix(self,T):
        return Ptensors0_fromMxFn.apply(T)

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

    def __getitem__(self,i):
        return Ptensors0_getFn.apply(self,i)
    
    def torch(self):
        return Ptensors0_toMxFn.apply(self)
    

    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        return Ptensors0_addFn.apply(self,y)

    def __mul__(self,y):
        return Ptensors0_mprodFn.apply(self,y)

    def concat(self,y):
        return Ptensors0_concatFn.apply(self,y)
    
    def linmaps0(self):
        return Ptensors0_Linmaps0Fn.apply(self);

    def linmaps1(self):
        return Ptensors0_Linmaps1Fn.apply(self);

    def linmaps2(self):
        return Ptensors0_Linmaps2Fn.apply(self);


    def transfer0(self,_atoms):
        return Ptensors0_Transfer0Fn.apply(self,_atoms)

    def transfer1(self,_atoms):
        return Ptensors0_Transfer1Fn.apply(self,_atoms)

    def transfer2(self,_atoms):
        return Ptensors0_Transfer2Fn.apply(self,_atoms)

    
    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------


class Ptensors0_fromMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptensors0(1)
        R.obj=_ptensors0(x)
        return R

    @staticmethod
    def backward(ctx,g):
        return g.obj.torch()


class Ptensors0_toMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        return x.obj.torch()
 
    @staticmethod
    def backward(ctx,g):
       R=ptensors0(1)
       R.obj=_ptensors0(x)
       return R
    

class Ptensors0_getFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,i):
        R=ptens.ptensor0(x.obj[i].torch())
        R.atoms=x.atoms_of(i)
        ctx.x=x.obj
        ctx.i=i
        return R

    @staticmethod
    def backward(ctx,g):
        R=ptensors0(1)
        ctx.x.add_to_grad(ctx.i,g)
        return R,None


class Ptensors0_addFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptensors0(1)
        R.obj=_ptensors0(x.obj)
        R.obj.add(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.r.get_gradp())
        ctx.y.add_to_grad(ctx.r.get_gradp())
        return ptensors0(1),ptensors0(1)

class Ptensors0_concatFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=ptensors0(1)
        r.obj=_ptensors0.concat(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_concat_back(ctx.r,0)
        ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
        return ptensors0(1),ptensors0(1)

    
class Ptensors0_mprodFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),y.size(1))
        R.obj.add_mprod(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
        return ptensors0(1), ctx.x.mprod_back1(ctx.r.gradp())


class Ptensors0_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc())
        ptens_base.add_linmaps0to0(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps0to0_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors0(1)


class Ptensors0_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),x.obj.get_nc())
        ptens_base.add_linmaps0to1(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps0to1_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors0(1)


class Ptensors0_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*2)
        ptens_base.add_linmaps0to2(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps0to2_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors0(1)


