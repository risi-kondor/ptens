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
from typing import Literal

import torch

import ptens_base
from ptens_base import nodelayer as _nodelayer
from ptens.utility import device_id as device_id

import ptens.ptensor0
import ptens.ptensors1
import ptens.ptensors2 

from ptens.subgraphlayer import *


class nodelayer(torch.Tensor):

    @classmethod
    def from_matrix(self,G,T):
        return Nodelayer_fromMxFn.apply(G,T)
            
    @classmethod
    def dummy(self):
        R=nodelayer(1)
        #R.obj=_nodelayer.dummy()
        return R

    @classmethod
    def raw(self,G, _nc, device='cpu'):
        R=nodelayer(1)
        R.obj=_nodelayer.raw(G.obj,_nc,ptens.device_id(device))
        return R

    @classmethod
    def zeros(self,G, _nc, device='cpu'):
        R=nodelayer(1)
        R.obj=_nodelayer.zero(G.obj,_nc,ptens.device_id(device))
        return R

    @classmethod
    def randn(self,G, _nc, device='cpu'):
        R=nodelayer(1)
        R.obj=_nodelayer.gaussian(G.obj,_nc,ptens.device_id(device))
        return R

    @classmethod
    def sequential(self,G, _nc, device='cpu'):
        R=nodelayer(1)
        R.obj=_nodelayer.sequential(G.obj,_nc,ptens.device_id(device))
        return R

    @classmethod
    def gather(self,x):
        return Nodelayer_GatherFn.apply(x)

    @classmethod
    def gather_from_ptensors(self,x,G):
        return Nodelayer_GatherFromPtensorsFn.apply(x,G)


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=nodelayer(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=nodelayer(1)
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

    def push_back(self, x):
        return self.obj.push_back(x)

    def __getitem__(self,i):
        return Nodelayer_getFn.apply(self,i)
    
    def torch(self):
        return Nodelayer_toMxFn.apply(self)

    def to(self, device='cpu'):
        return Nodelayer_toDeviceFn.apply(self,device)
        

    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        return Nodelayer_plusFn.apply(self,y)

    def __mul__(self,y):
        return Nodelayer_mprodFn.apply(self,y)

    def linear(self,y,b):
        return Nodelayer_linearFn.apply(self,y,b)

    def concat(self,y):
        return Nodelayer_concatFn.apply(self,y)

    def relu(self,alpha=0.5):
        return Nodelayer_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return Nodelayer_inpFn.apply(self,y)
    
    def diff2(self,y):
        return Nodelayer_diff2Fn.apply(self,y)

    def scale(self,y):
        return Nodelayer_scaleFn.apply(self,y)

    def mult_channels(self,y):
        return Nodelayer_mult_channelsFn.apply(self,y)

    def normalize_channels(self):
        return Nodelayer_normalize_channels.apply(self)
    

    # ---- Message passing -----------------------------------------------------------------------------------
    

#     def linmaps0(self,normalized=False):
#         return Nodelayer_Linmaps0Fn.apply(self);

#     def linmaps1(self,normalized=False):
#         return Nodelayer_Linmaps1Fn.apply(self);

#     def linmaps2(self,normalized=False):
#         return Nodelayer_Linmaps2Fn.apply(self);


    def transfer0(self,_atoms,G,normalized=False):
        return Nodelayer_Transfer0Fn.apply(self,_atoms,G)

    def transfer1(self,_atoms,G,normalized=False):
        return Nodelayer_Transfer1Fn.apply(self,_atoms,G)

    def transfer2(self,_atoms,G,normalized=False):
        return Nodelayer_Transfer2Fn.apply(self,_atoms,G)
    

    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------


class Nodelayer_fromMxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,G,x):
        R=nodelayer(1)
        R.obj=_nodelayer(G.obj,x)
        return R
    @staticmethod
    def backward(ctx,g):
        return None, ctx.r.get_grad().torch()


class Nodelayer_toMxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.x=x.obj
        return x.obj.torch()
    @staticmethod
    def backward(ctx,g):
       R=nodelayer(1)
       ctx.x.torch_back(g)
       return R
    

# class Nodelayer_getFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,i):
#         R=ptens.ptensor0(x.obj[i].torch())
#         R.atoms=x.atoms_of(i)
#         ctx.x=x.obj
#         ctx.i=i
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         R=nodelayer(1)
#         ctx.x.add_to_grad(ctx.i,g)
#         return R,None


# class Nodelayer_toFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,_dev):
#         dev=ptens.device_id(_dev)
#         R=nodelayer(1)
#         R.obj=_nodelayer(x.obj,dev)
#         ctx.x=x.obj
#         ctx.r=R.obj
#         ctx.dev=dev
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.to_device_back(ctx.r,ctx.dev)
#         return nodelayer.dummy(), None
        

# class Nodelayer_addFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         R=nodelayer(1)
#         R.obj=_nodelayer(x.obj)
#         R.obj.add(y.obj)
#         ctx.x=x.obj
#         ctx.y=y.obj
#         ctx.r=R.obj
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_to_grad(ctx.r.get_gradp())
#         ctx.y.add_to_grad(ctx.r.get_gradp())
#         return nodelayer(1),nodelayer(1)


# class Nodelayer_inpFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         ctx.x=x.obj
#         ctx.y=y.obj
#         return torch.tensor(x.obj.inp(y.obj))
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_to_grad(ctx.y,g.item())
#         ctx.y.add_to_grad(ctx.x,g.item())
#         return nodelayer.dummy(), nodelayer.dummy()


# class Nodelayer_diff2Fn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         ctx.x=x.obj
#         ctx.y=y.obj
#         return torch.tensor(x.obj.diff2(y.obj))
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_to_grad(ctx.x,g.item()*2.0)
#         ctx.x.add_to_grad(ctx.y,-g.item()*2.0)
#         ctx.y.add_to_grad(ctx.y,g.item()*2.0)
#         ctx.y.add_to_grad(ctx.x,-g.item()*2.0)
#         return nodelayer.dummy(), nodelayer.dummy()


# class Nodelayer_concatFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         r=nodelayer(1)
#         r.obj=_nodelayer.concat(x.obj,y.obj)
#         ctx.x=x.obj
#         ctx.y=y.obj
#         ctx.r=r.obj
#         return r
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_concat_back(ctx.r,0)
#         ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
#         return nodelayer(1),nodelayer(1)

    
# class Nodelayer_mprodFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         R=ptens.nodelayer.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
#         R.obj.add_mprod(x.obj,y)
#         ctx.x=x.obj
#         ctx.y=y
#         ctx.r=R.obj
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
#        return nodelayer.dummy(), ctx.x.mprod_back1(ctx.r.gradp())


class Nodelayer_mult_channelsFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.nodelayer.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale_channels(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_channels_back0(ctx.r.gradp(),ctx.y)
        return nodelayer.dummy(), None


class Nodelayer_scaleFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.nodelayer.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_back0(ctx.r.gradp(),ctx.y)
        return nodelayer.dummy(), ctx.x.scale_back1(ctx.r.gradp())

class Nodelayer_normalize_channels(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=nodelayer(1)
        r.obj=x.obj.normalize_columns()
        ctx.x=x.obj
        ctx.r=r.obj
        return r;

    @staticmethod
    def backward(ctx,g):
        ctx.x.normalize_columns_back(ctx.r)
        return nodelayer.dummy()
    




class Nodelayer_GatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        r=ptens.nodelayer(1)
        r.obj=_nodelayer(x.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.r.gather_back(ctx.x)
        return nodelayer.dummy(), None


class Nodelayer_GatherFromPtensorsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,G):
        r=ptens.nodelayer(1)
        r.obj=_nodelayer(G.obj,x.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.r.gather_back(ctx.x)
        return nodelayer.dummy(), None, None


class Nodelayer_Outer0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptens.nodelayer(1)
        r.obj=ptens_base.outer(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return nodelayer.dummy(), nodelayer.dummy()


class Nodelayer_Outer1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptens.ptensors1(1)
        r.obj=ptens_base.outer(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return nodelayer.dummy(), ptens.ptensors1.dummy()


class Nodelayer_Outer2Fn(torch.autograd.Function):

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
        return nodelayer.dummy(), ptens.ptensors2.dummy()






