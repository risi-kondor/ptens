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
from ptens_base import ptensorlayer as _tlayer
from ptens.utility import device_id as device_id

#import ptens.ptensor0
#import ptens.ptensors1
#import ptens.ptensors2 


class tlayer(torch.Tensor):

    @classmethod
    def dummy(self):
        R=tlayer(1)
        return R

    @classmethod
    def init(self,obj):
        R=tlayer(1)
        R.obj=obj
        return R
    
    @classmethod
    def from_matrix(self,k,M,atoms=None):
        return PtensorLayer_fromMxFn.apply(k,M,atoms)
            
    @classmethod
    def zeros(self, k, _atoms, _nc, device='cpu'):
        R=tlayer(1)
        R.obj=_tlayer.create(k,_atoms,_nc,0,device_id(device))
        return R

    @classmethod
    def randn(self, k, _atoms, _nc, device='cpu'):
        R=tlayer(1)
        R.obj=_tlayer.create(k,_atoms,_nc,4,device_id(device))
        return R

    @classmethod
    def sequential(self, k, _atoms, _nc, device='cpu'):
        R=tlayer(1)
        R.obj=_tlayer.create(k,_atoms,_nc,3,device_id(device))
        return R

    def randn_like(self):
        return tlayer.init(self.obj.randn_like())
    
    @classmethod
    def cat(self,*args):
        return Ptensors1_catFn.apply(self,*args)


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()

    def get_grad(self):
        R=tlayer(1)
        R.obj=self.obj.get_grad()
        return R
    
    def get_dev(self):
        return self.obj.get_dev()

    def get_nc(self):
        return self.obj.get_nc()

    def get_atoms(self):
        return self.obj.get_atoms()
    
    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def torch(self):
        return PtensorLayer_toMxFn.apply(self)
        
    def to(self, device='cpu'):
        return PtensorLayer_toFn.apply(self,device)


    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        return PtensorLayer_addFn.apply(self,y)

    def cat_channels(self,y):
        return PtensorLayer_cat_channelsFn.apply(self,y)

    def outer(self,y):
        return PtensorLayer_outerFn.apply(self,y)

    def __mul__(self,y):
        return PtensorLayer_mprodFn.apply(self,y)

    def linear(self,y,b):
        return PtensorLayer_linearFn.apply(self,y,b)

    def scale(self,y):
        return PtensorLayer_scaleFn.apply(self,y)

    def avg_pool(self):
        return PtensorLayer_averageFn.apply(self)

    def mult_channels(self,y):
        return PtensorLayer_mult_channelsFn.apply(self,y)

    def relu(self,alpha=0.1):
        return PtensorLayer_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return PtensorLayer_inpFn.apply(self,y)
    
    def diff2(self,y):
        return PtensorLayer_diff2Fn.apply(self,y)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------



# ----- Transport and conversions ----------------------------------------------------------------------------


class PtensorLayer_fromMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,k,x,atoms=None):
        R=tlayer(1)
        if atoms is None:
            R.obj=_tlayer(k,x)
        else:
            R.obj=_tlayer(k,atoms,x)
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        return ctx.r.get_grad().torch(), None


class PtensorLayer_toMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x.obj
        return x.obj.torch()

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(g)
        return tlayer.dummy()


class PtensorLayer_toFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,_dev):
        dev=ptens.device_id(_dev)
        r=x.dummy()
        r.obj=x.obj.to_device(dev)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.dev=dev
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.to_device_back(ctx.r,ctx.dev)
        return tlayer.dummy(), None
        

# ---- Arithmetic --------------------------------------------------------------------------------------------


class PtensorLayer_addFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=x.dummy()
        r.obj=x.obj.copy()
        r.obj.add(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_back(ctx.r)
        ctx.y.add_back(ctx.r)
        return tlayer.dummy(),tlayer.dummy()


class PtensorLayer_cat_channelsFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=x.dummy(1)
        r.obj=x.obj.cat_channels(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.cat_channels_back0(ctx.r)
        ctx.y.cat_channels_back1(ctx.r)
        return tlayer.dummy(),tlayer.dummy()


class PtensorLayer_catFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,dummy,*args):
        r=tlayer.dummy()
        ctx.args=[x.obj for x in args]
        r.obj=_tlayer.cat(ctx.args)
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        offs=0
        dummies=[]
        for x in ctx.args:
            x.add_cat_back(ctx.r,offs)
            offs=offs+x.dim(0)
            dummies.append(tlayer.dummy())
        return None, *dummies


class PtensorLayer_outerFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=tlayer.dummy()
        r.obj=x.obj.outer(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ctx.x.outer_back0(ctx.r,ctx.y)
        ctx.y.outer_back0(ctx.r,ctxxy)
        return tlayer.dummy(), tlayer.dummy()


class PtensorLayer_mprodFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=x.dummy()
        r.obj=x.obj.mprod(y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_mprod_back0(ctx.r,ctx.y)
        return tlayer.dummy(), ctx.x.mprod_back1(ctx.r)


class PtensorLayer_linearFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y,b):
        r=x.dummy()
        r.obj=x.obj.linear(y,b)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linear_back0(ctx.r,ctx.y)
        return tlayer.dummy(), ctx.x.linear_back1(ctx.r), ctx.x.linear_back2(ctx.r)


class PtensorLayer_scaleFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=x.dummy()
        r.obj=x.obj.scale(y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_back0(ctx.r,ctx.y)
        return tlayer.dummy(), ctx.x.scale_back1(ctx.r)


class PtensorLayer_mult_channelsFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=x.dummy()
        r.obj=x.obj.mult_channels(y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_mult_channels_back0(ctx.r,ctx.y)
        return tlayer.dummy(), None


class PtensorLayer_ReLUFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,alpha):
        r=x.dummy()
        r.obj=x.obj.ReLU(alpha)
        ctx.x=x.obj
        ctx.alpha=alpha
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_ReLU_back(ctx.r,ctx.x,ctx.alpha)
        return tlayer.dummy(), None


class PtensorLayer_inpFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x.obj
        ctx.y=y.obj
        return torch.tensor(x.obj.inp(y.obj))

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.y,g.item())
        ctx.y.add_to_grad(ctx.x,g.item())
        return tlayer.dummy(), tlayer.dummy()


class PtensorLayer_diff2Fn(torch.autograd.Function):
    
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
        return tlayer.dummy(), tlayer.dummy()


