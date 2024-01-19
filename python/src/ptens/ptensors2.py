#
# This file is part of ptens, a C++/CUDA library for permutation 
# equivariant message passing. 
#  
# Copyright (c) 2023, Imre Risi Kondor
#
# This source code file is subject to the terms of the noncommercial 
# license distributed with cnine in the file LICENSE.TXT. Commercial 
# use is prohibited. All redistributed versions of this file (in 
# original or modified form) must retain this copyright notice and 
# must be accompanied by a verbatim copy of the license. 
#
#
import torch

import ptens_base
from ptens_base import ptensors2 as _ptensors2
from ptens.utility import device_id as device_id

import ptens.ptensors0 
import ptens.ptensors1 


class ptensors2(torch.Tensor):

    @classmethod
    def from_matrix(self,T,atoms):
        return Ptensors2_fromMxFn.apply(T,atoms)

    @classmethod
    def dummy(self):
        R=ptensors2(1)
        R.obj=_ptensors2.dummy()
        return R

    @classmethod
    def raw(self, _atoms, _nc, device=0):
        R=ptensors2(1)
        R.obj=_ptensors2.raw(_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def zeros(self, _atoms, _nc, device=0):
        R=ptensors2(1)
        R.obj=_ptensors2.zero(_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def randn(self, _atoms, _nc, _sigma=1.0, device=0):
        R=ptensors2(1)
        R.obj=_ptensors2.gaussian(_atoms,_nc,_sigma,ptens.device_id(device))
        return R

    @classmethod
    def sequential(self, _atoms, _nc, device=0):
        R=ptensors2(1)
        R.obj=_ptensors2.sequential(_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def cat(self,*args):
        return Ptensors2_catFn.apply(self,*args)

    @classmethod
    def sum(self,*args):
        return Ptensors2_sumFn.apply(self,*args)

    def randn_like(self,sigma=1.0):
        return ptensors2.randn(self.get_atoms(),self.get_nc(),sigma,self.get_dev())


    # ----- Access -------------------------------------------------------------------------------------------


    def __len__(self):
        return len(self.obj)
    
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


    def get_dev(self):
        return self.obj.get_dev()

    def get_nc(self):
        return self.obj.get_nc()

    def get_atoms(self):
        return self.obj.get_atoms()
    
    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def __getitem__(self,i):
        return Ptensors2_getFn.apply(self,i)
    
    def push_back(self, x):
        return self.obj.push_back(x)

    def torch(self):
        return Ptensors2_toMxFn.apply(self)

    def to(self, device='cpu'):
        return Ptensors2_toFn.apply(self,device)
        #self.obj.to_device(ptens.device_id(_device))


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def __add__(self,y):
        return Ptensors2_addFn.apply(self,y)

    def __mul__(self,y):
        return Ptensors2_mprodFn.apply(self,y)

    def linear(self,y,b):
        return Ptensors2_linearFn.apply(self,y,b)

    def concat(self,y):
        return Ptensors2_concatFn.apply(self,y)

    def relu(self,alpha=0.5):
        return Ptensors2_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return Ptensors2_inpFn.apply(self,y)
    
    def diff2(self,y):
        return ptens.Ptensors2_diff2Fn.apply(self,y)

    def outer(self,y):
        if isinstance(y,ptens.ptensors0):
            return Ptensors2_Outer0Fn.apply(self,y)

    def scale(self,y):
        return Ptensors2_scaleFn.apply(self,y)

    def mult_channels(self,y):
        return Ptensors2_mult_channelsFn.apply(self,y)


    # ---- Message passing -----------------------------------------------------------------------------------


    def linmaps0(self,normalized=False):
        return Ptensors2_Linmaps0Fn.apply(self,normalized);

    def linmaps1(self,normalized=False):
        return Ptensors2_Linmaps1Fn.apply(self,normalized);

    def linmaps2(self,normalized=False):
        return Ptensors2_Linmaps2Fn.apply(self,normalized);


    def transfer0(self,_atoms,normalized=False):
        return Ptensors2_Transfer0Fn.apply(self,_atoms,normalized)

    def transfer1(self,_atoms,normalized=False):
        return Ptensors2_Transfer1Fn.apply(self,_atoms,normalized)

    def transfer2(self,_atoms,normalized=False):
        return Ptensors2_Transfer2Fn.apply(self,_atoms,normalized)


    def unite1(self,G,normalized=False):
        return Ptensors2_Unite1Fn.apply(self,G,normalized)
    
    def unite2(self,G,normalized=False):
        return Ptensors2_Unite2Fn.apply(self,G,normalized)
    

    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------


class Ptensors2_fromMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms):
        R=ptensors2(1)
        R.obj=_ptensors2(x,atoms)
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        return ctx.r.get_grad().torch(), None


class Ptensors2_toMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x.obj
        return x.obj.torch()
 
    @staticmethod
    def backward(ctx,g):
       R=ptensors2(1)
       ctx.x.add_to_grad(_ptensors2(g,ctx.x.get_atoms()))
       return R
    

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


class Ptensors2_toFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,_dev):
        dev=ptens.device_id(_dev)
        R=ptensors2(1)
        R.obj=_ptensors2(x.obj,dev)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.dev=dev
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.move_to_device_back(ctx.r.get_gradp(),ctx.dev)
        return ptensors2.dummy(), None
        

class Ptensors2_addFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptensors2(1)
        R.obj=_ptensors2(x.obj)
        R.obj.add(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.r.get_gradp())
        ctx.y.add_to_grad(ctx.r.get_gradp())
        return ptensors2(1),ptensors2(1)


class Ptensors2_ReLUFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,alpha):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_ReLU(x.obj,alpha)
        ctx.x=x.obj
        ctx.alpha=alpha
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_ReLU_back(ctx.r.gradp(),ctx.alpha)
        return ptensors2.dummy(), None


class Ptensors2_inpFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x.obj
        ctx.y=y.obj
        return torch.tensor(x.obj.inp(y.obj))

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.y,g.item())
        ctx.y.add_to_grad(ctx.x,g.item())
        return ptensors2.dummy(), ptensors2.dummy()


class Ptensors0_diff2Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x.obj
        ctx.y=y.obj
        return torch.tensor(x.obj.diff2(y.obj))

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.x,g.item()*2.0)
        ctx.x.add_to_grad(ctx.y,-g.item()*2.0)
        ctx.y.add_to_grad(ctx.y,g.item()*2.0)
        ctx.y.add_to_grad(ctx.x,-g.item()*2.0)
        return ptensors2.dummy(), ptensors2.dummy()


class Ptensors2_concatFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=ptensors2(1)
        r.obj=_ptensors2.concat(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_concat_back(ctx.r,0)
        ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
        return ptensors2(1),ptensors2(1)


class Ptensors2_catFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,dummy,*args):
        r=ptensors2(1)
        ctx.args=[x.obj for x in args]
        r.obj=_ptensors2.cat(ctx.args)
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        offs=0
        dummies=[]
        for x in ctx.args:
            x.add_cat_back(ctx.r,offs)
            offs=offs+len(x)
            dummies.append(ptensors1(1))
        return None, *dummies


class Ptensors0_sumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,dummy,*args):
        r=ptensors2(1)
        ctx.args=[x.obj for x in args]
        r.obj=_ptensors2.sum(ctx.args)
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        dummies=[]
        for x in ctx.args:
            x.add_to_grad(ctx.r.get_gradp())
            dummies.append(ptensors2(1))
        return None, *dummies


class Ptensors2_mprodFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_mprod(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
        return ptens.ptensors1(2), ctx.x.mprod_back1(ctx.r.gradp())


class Ptensors2_scaleFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_back0(ctx.r.gradp(),ctx.y)
        return ptensors2.dummy(), ctx.x.scale_back1(ctx.r.gradp())


class Ptensors2_mult_channelsFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale_channels(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_channels_back0(ctx.r.gradp(),ctx.y)
        return ptensors2.dummy(), None


class Ptensors2_linearFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y,b):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_linear(x.obj,y,b)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linear_back0(ctx.r.gradp(),ctx.y)
        return ptensors2.dummy(), ctx.x.linear_back1(ctx.r.gradp()), ctx.x.linear_back2(ctx.r.gradp())


class Ptensors2_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,normalized=False):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*2,x.obj.get_dev())
        if(normalized):
            ptens_base.add_linmaps2to0_n(R.obj,x.obj)
        else:
            ptens_base.add_linmaps2to0(R.obj,x.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.add_linmaps2to0_back_n(ctx.x.gradp(),ctx.r.gradp())
        else:
            ptens_base.add_linmaps2to0_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors2(1), None


class Ptensors2_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,normalized=False):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*5,x.obj.get_dev())
        if(normalized):
            ptens_base.add_linmaps2to1_n(R.obj,x.obj)
        else:
            ptens_base.add_linmaps2to1(R.obj,x.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.add_linmaps2to1_back_n(ctx.x.gradp(),ctx.r.gradp())
        else:
            ptens_base.add_linmaps2to1_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors2(1), None


class Ptensors2_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,normalized=False):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*15,x.obj.get_dev())
        if(normalized):
            ptens_base.add_linmaps2to2_n(R.obj,x.obj)
        else:
            ptens_base.add_linmaps2to2(R.obj,x.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.add_linmaps2to2_back_n(ctx.x.gradp(),ctx.r.gradp())
        else:
            ptens_base.add_linmaps2to2_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors2(1), None


class Ptensors2_Transfer0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G,normalized=False):
        ctx.x=x
        R=ptens.ptensors0.zeros(atoms,x.obj.get_nc()*2,x.obj.get_dev())
        if(normalized):
            ptens_base.add_msg_n(R.obj,x.obj,G.obj)
        else:   
            ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.G=G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.add_msg_back_n(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        else:
            ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors2.dummy(), None, None, None


class Ptensors2_Transfer1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G,normalized=False):
        ctx.x=x
        R=ptens.ptensors1.zeros(atoms,x.obj.get_nc()*5,x.obj.get_dev())
        if(normalized):
            ptens_base.add_msg_n(R.obj,x.obj,G.obj)
        else:
            ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.G=G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.add_msg_back_n(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        else:
            ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors2.dummy(), None, None, None


class Ptensors2_Transfer2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G,normalized=False):
        ctx.x=x
        R=ptens.ptensors2.zeros(atoms,x.obj.get_nc()*15,x.obj.get_dev())
        if(normalized):
            ptens_base.add_msg_n(R.obj,x.obj,G.obj)
        else:
            ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.G=G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.add_msg_back_n(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        else:
            ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors2.dummy(), None, None, None


class Ptensors2_Unite1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G,normalized=False):
        r=ptens.ptensors1(1)
        if(normalized):
            r.obj=ptens_base.unite1_n(x.obj,G.obj)
        else:
            r.obj=ptens_base.unite1(x.obj,G.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.G=G.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.unite2to1_back_n(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        else:
            ptens_base.unite2to1_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors2.dummy(), None, None


class Ptensors2_Unite2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G,normalized=False):
        r=ptens.ptensors2(1)
        if(normalized):
            r.obj=ptens_base.unite2_n(x.obj,G.obj)
        else:
            r.obj=ptens_base.unite2(x.obj,G.obj)
        ctx.normalized=normalized
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.G=G.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        if(ctx.normalized):
            ptens_base.unite2to2_back_n(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        else:
            ptens_base.unite2to2_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors2.dummy(), None, None


class Ptensors2_Outer0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptens.ptensors2(1)
        r.obj=ptens_base.outer(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return ptensors2.dummy(), ptens.ptensors0.dummy()





