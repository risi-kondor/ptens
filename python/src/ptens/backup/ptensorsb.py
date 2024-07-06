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
from ptens_base import ptensors0b as _ptensors0b
from ptens_base import ptensors1b as _ptensors1b
from ptens_base import ptensors2b as _ptensors2b
from ptens.utility import device_id as device_id
#from ptens.ptensors0b import * 

#import ptens.ptensor0
#import ptens.ptensors1
#import ptens.ptensors2 


class ptensorsb(torch.Tensor):

    @classmethod
    def dummy(self):
        R=ptensorsb(1)
        return R

    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()

    def get_dev(self):
        return self.obj.get_dev()

    def get_nc(self):
        return self.obj.get_nc()

    def get_atoms(self):
        return self.obj.get_atoms()
    
    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def torch(self):
        return Ptensorsb_toMxFn.apply(self)
        
    def to(self, device='cpu'):
        return Ptensorsb_toFn.apply(self,device)


    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        return Ptensorsb_addFn.apply(self,y)

    def __mul__(self,y):
        return Ptensorsb_mprodFn.apply(self,y)

    def linear(self,y,b):
        return Ptensorsb_linearFn.apply(self,y,b)

    def scale(self,y):
        return Ptensorsb_scaleFn.apply(self,y)

    def avg_pool(self):
        return Ptensorsb_averageFn.apply(self)

    def mult_channels(self,y):
        return Ptensorsb_mult_channelsFn.apply(self,y)

    def relu(self,alpha=0.5):
        return Ptensorsb_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return Ptensorsb_inpFn.apply(self,y)
    
    def diff2(self,y):
        return Ptensorsb_diff2Fn.apply(self,y)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------



# ----- Transport and conversions ----------------------------------------------------------------------------


class Ptensorsb_likeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,M):
        r=x.dummy()
        r.obj=x.obj.like(M)
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        return None, ctx.r.get_grad().torch()


class Ptensorsb_toMxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.x=x.obj
        return x.obj.torch()

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(g)
        return ptensorsb.dummy()


class Ptensorsb_toFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,_dev):
        dev=device_id(_dev)
        r=x.dummy()
        r.obj=x.obj.to(dev)
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.dev=dev
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.to_device_back(ctx.r,ctx.dev)
        return ptensorsb.dummy(), None
        

# ---- Arithmetic --------------------------------------------------------------------------------------------


class Ptensorsb_addFn(torch.autograd.Function):
    
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
        return ptensorsb.dummy(),ptensorsb.dummy()


class Ptensorsb_cat_channelsFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=x.dummy()
        r.obj=x.obj.cat_channels(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.cat_channels_back0(ctx.r)
        ctx.y.cat_channels_back1(ctx.r)
        return ptensorsb.dummy(),ptensorsb.dummy()


# class Ptensorsb_catFn(torch.autograd.Function):
    
#     @staticmethod
#     def forward(ctx,dummy,*args):
#         r=args[0].dummy()
#         ctx.args=[x.obj for x in args]
#         r.obj=args[0].obj.cat(ctx.args)
#         ctx.r=r.obj
#         return r

#     @staticmethod
#     def backward(ctx,g):
#         offs=0
#         dummies=[]
#         for x in ctx.args:
#             x.add_cat_back(ctx.r,offs)
#             offs=offs+len(x)
#             dummies.append(ptensorsb(1))
#         return None, *dummies


class Ptensorsb_mprodFn(torch.autograd.Function):
    
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
        return ptensorsb.dummy(), ctx.x.mprod_back1(ctx.r)


class Ptensorsb_linearFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y,b):
        r=x.dummy()
        r.obj=x.obj.linear(y,b)
        ctx.x=x.obj
        ctx.y=y
        ctx.b=b
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linear_back0(ctx.r,ctx.y)
        return ptensorsb.dummy(), ctx.x.linear_back1(ctx.r), ctx.x.linear_back2(ctx.r)


class Ptensorsb_scaleFn(torch.autograd.Function):
    
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
        return ptensorsb.dummy(), ctx.x.scale_back1(ctx.r)


class Ptensorsb_mult_channelsFn(torch.autograd.Function):
    
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
        return ptensorsb.dummy(), None


class Ptensorsb_ReLUFn(torch.autograd.Function):
    
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
        ctx.x.add_ReLU_back(ctx.r,ctx.alpha)
        return ptensorsb.dummy(), None


class Ptensorsb_inpFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x.obj
        ctx.y=y.obj
        return torch.tensor(x.obj.inp(y.obj))

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.y,g.item())
        ctx.y.add_to_grad(ctx.x,g.item())
        return ptensorsb.dummy(), ptensorsb.dummy()


class Ptensorsb_diff2Fn(torch.autograd.Function):
    
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
        return ptensorsb.dummy(), ptensorsb.dummy()


# ---- Message passing --------------------------------------------------------------------------------------


class Ptensorsb_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=x.dummy()
        r.obj=_ptensors0b.linmaps(x.obj) 
        ctx.x=x.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linmaps_back(ctx.r)
        return ptensorsb.dummy()


class Ptensorsb_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=x.dummy()
        r.obj=_ptensors1b.linmaps(x.obj) 
        ctx.x=x.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linmaps_back(ctx.r)
        return ptensorsb.dummy()


class Ptensorsb_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=x.dummy()
        r.obj=_ptensors2b.linmaps(x.obj) 
        ctx.x=x.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linmaps_back(ctx.r)
        return ptensorsb.dummy()


class Ptensorsb_Gather0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms):
        r=x.dummy()
        r.obj=_ptensors0b.gather(x.obj,atoms)
#        r.obj=x.obj.gather0(atoms)
        ctx.x=x.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_gather_back(ctx.r)
        return ptensorsb.dummy(), None 


class Ptensorsb_Gather1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,min_overlaps):
        r=x.dummy()
        r.obj=_ptensors1b.gather(x.obj,atoms,min_overlaps)
        ctx.x=x.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_gather_back_alt(ctx.r)
        return ptensorsb.dummy(), None, None


class Ptensorsb_Gather2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms):
        r=x.dummy()
        r.obj=_ptensors2b.gather(x.obj,atoms)
        ctx.x=x.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_gather_back(ctx.r)
        return ptensorsb.dummy(), None


