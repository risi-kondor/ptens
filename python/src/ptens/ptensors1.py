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

    def __getitem__(self,i):
        return Ptensors1_getFn.apply(self,i)
    
    def push_back(self, x):
        return self.obj.push_back(x)


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def __add__(self,y):
        return Ptensors1_addFn.apply(self,y)

    def __mul__(self,y):
        return Ptensors1_mprodFn.apply(self,y)

    def concat(self,y):
        return Ptensors1_concatFn.apply(self,y)
    

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


class Ptensors1_getFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,i):
        R=ptens.ptensor1(x.obj[i].torch())
        R.atoms=x.atoms_of(i)
        ctx.x=x.obj
        ctx.i=i
        return R

    @staticmethod
    def backward(ctx,g):
        R=ptensors1(1)
        ctx.x.add_to_grad(ctx.i,g)
        return R,None

    
class Ptensors1_addFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptensors1(1)
        R.obj=_ptensors1(x.obj)
        R.obj.add(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.r.get_gradp())
        ctx.y.add_to_grad(ctx.r.get_gradp())
        return ptensors1(1),ptensors1(1)


class Ptensors1_concatFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=ptensors1(1)
        r.obj=_ptensors1.concat(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_concat_back(ctx.r,0)
        ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
        return ptensors1(1),ptensors1(1)


class Ptensors1_mprodFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),y.size(1))
        R.obj.add_mprod(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
        return ptensors1(1), ctx.x.mprod_back1(ctx.r.gradp())


class Ptensors1_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc())
        ptens_base.add_linmaps1to0(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps1to0_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors1(1)


class Ptensors1_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*2)
        ptens_base.add_linmaps1to1(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps1to1_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors1(1)


class Ptensors1_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*5)
        ptens_base.add_linmaps1to2(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps1to2_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors1(1)







