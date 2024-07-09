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
from ptens_base import subgraph_layer0 as _subgraph_layer0
from ptens.utility import device_id as device_id

import ptens.ptensor0
import ptens.ptensors1
import ptens.ptensors2 

from ptens.subgraphlayer import *


class subgraphlayer0(subgraphlayer):

    @classmethod
    def from_matrix(self,G,T):
        return SubgraphLayer0_fromMxFn.apply(G,T)
            
    @classmethod
    def dummy(self):
        R=subgraphlayer0(1)
        R.obj=_subgraph_layer0.dummy()
        return R

    @classmethod
    def raw(self,G, _nc, device='cpu'):
        R=subgraphlayer0(1)
        R.obj=_subgraph_layer0.raw(G.obj,_nc,ptens.device_id(device))
        return R

    @classmethod
    def zeros(self,G, _nc, device='cpu'):
        R=subgraphlayer0(1)
        R.obj=_subgraph_layer0.zero(G.obj,_nc,ptens.device_id(device))
        return R

    @classmethod
    def randn(self,G, _nc, device='cpu'):
        R=subgraphlayer0(1)
        R.obj=_subgraph_layer0.gaussian(G.obj,_nc,ptens.device_id(device))
        return R

    @classmethod
    def sequential(self,G, _nc, device='cpu'):
        R=subgraphlayer0(1)
        R.obj=_subgraph_layer0.sequential(G.obj,_nc,ptens.device_id(device))
        return R


    #def randn_like(self,sigma=1.0):
     #   return subgraphlayer0.randn(self.get_atoms(),self.get_nc(),sigma,self.get_dev())


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=subgraph_layer0(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=subgraphlayer0(1)
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
        return Subgraph_layer0_getFn.apply(self,i)
    
    def torch(self):
        return SubgraphLayer0_toMxFn.apply(self)

    def to(self, device='cpu'):
        return Subgraph_layer0_toFn.apply(self,device)
        #self.obj.to_device(ptens.device_id(device))
        

    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        return SubgraphLayer0_addFn.apply(self,y)

    def __mul__(self,y):
        return SubgraphLayer0_mprodFn.apply(self,y)

    def linear(self,y,b):
        return SubgraphLayer0_linearFn.apply(self,y,b)

    def concat(self,y):
        return SubgraphLayer0_concatFn.apply(self,y)

    def relu(self,alpha=0.5):
        return SubgraphLayer0_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return SubgraphLayer_inpFn.apply(self,y)
    
    def diff2(self,y):
        return SubgraphLayer0_diff2Fn.apply(self,y)

    def outer(self,y):
        if isinstance(y,ptens.subgraphlayer0):
            return Subgraph_layer0_Outer0Fn.apply(self,y)
        if isinstance(y,ptens.ptensors1):
            return Subgraph_layer0_Outer1Fn.apply(self,y)
        if isinstance(y,ptens.ptensors2):
            return Subgraph_layer0_Outer2Fn.apply(self,y)

    def scale(self,y):
        return Subgraph_layer0_scaleFn.apply(self,y)

    def mult_channels(self,y):
        return Subgraph_layer0_mult_channelsFn.apply(self,y)

    def normalize_channels(self):
        return SubgraphLayer0_normalize_channels.apply(self)
    

    # ---- Message passing -----------------------------------------------------------------------------------
    

    def linmaps0(self,normalized=False):
        return Subgraph_layer0_Linmaps0Fn.apply(self);

    def linmaps1(self,normalized=False):
        return Subgraph_layer0_Linmaps1Fn.apply(self);

    def linmaps2(self,normalized=False):
        return Subgraph_layer0_Linmaps2Fn.apply(self);


    def transfer0(self,_atoms,G,normalized=False):
        return Subgraph_layer0_Transfer0Fn.apply(self,_atoms,G)

    def transfer1(self,_atoms,G,normalized=False):
        return Subgraph_layer0_Transfer1Fn.apply(self,_atoms,G)

    def transfer2(self,_atoms,G,normalized=False):
        return Subgraph_layer0_Transfer2Fn.apply(self,_atoms,G)


    def unite1(self,G,normalized=False):
        return Subgraph_layer0_Unite1Fn.apply(self,G)
    
    def unite2(self,G,normalized=False):
        return Subgraph_layer0_Unite2Fn.apply(self,G)
    
    #def gather(self,G,normalized=False):
    #    return Subgraph_layer0_GatherFn.apply(self,G)


    @classmethod
    def gather(self,x,S):
        return SubgraphLayer0_GatherFn.apply(x,S)
    

    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------


class SubgraphLayer0_fromMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,G,x):
        R=subgraphlayer0(1)
        R.obj=_subgraph_layer0(G.obj,x)
        return R

    @staticmethod
    def backward(ctx,g):
        return None, ctx.r.get_grad().torch()


class SubgraphLayer0_toMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x.obj
        return x.obj.torch()
 
    @staticmethod
    def backward(ctx,g):
       R=subgraphlayer0(1)
       ctx.x.torch_back(g)
       return R
    

class Subgraph_layer0_getFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,i):
        R=ptens.ptensor0(x.obj[i].torch())
        R.atoms=x.atoms_of(i)
        ctx.x=x.obj
        ctx.i=i
        return R

    @staticmethod
    def backward(ctx,g):
        R=subgraphlayer0(1)
        ctx.x.add_to_grad(ctx.i,g)
        return R,None


class SubgraphLayer0_toFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,_dev):
        dev=ptens.device_id(_dev)
        R=subgraphlayer0(1)
        R.obj=_subgraph_layer0(x.obj,dev)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.dev=dev
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.to_device_back(ctx.r,ctx.dev)
        return subgraphlayer0.dummy(), None
        

class SubgraphLayer0_addFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        R=subgraphlayer0(1)
        R.obj=_subgraph_layer0(x.obj)
        R.obj.add(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.r.get_gradp())
        ctx.y.add_to_grad(ctx.r.get_gradp())
        return subgraphlayer0(1),subgraphlayer0(1)


# class SubgraphLayer0_inpFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         ctx.x=x.obj
#         ctx.y=y.obj
#         return torch.tensor(x.obj.inp(y.obj))
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_to_grad(ctx.y,g.item())
#         ctx.y.add_to_grad(ctx.x,g.item())
#         return subgraphlayer0.dummy(), subgraphlayer0.dummy()


# class SubgraphLayer0_diff2Fn(torch.autograd.Function):
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
#         return subgraphlayer0.dummy(), subgraphlayer0.dummy()


class SubgraphLayer0_concatFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=subgraphlayer0(1)
        r.obj=_subgraph_layer0.concat(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_concat_back(ctx.r,0)
        ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
        return subgraphlayer0(1),subgraphlayer0(1)

    
class Subgraph_layer0_mprodFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.subgraphlayer0.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_mprod(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
        return subgraphlayer0.dummy(), ctx.x.mprod_back1(ctx.r.gradp())


class Subgraph_layer0_mult_channelsFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.subgraphlayer0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale_channels(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_channels_back0(ctx.r.gradp(),ctx.y)
        return subgraphlayer0.dummy(), None


class Subgraph_layer0_scaleFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.subgraphlayer0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_back0(ctx.r.gradp(),ctx.y)
        return subgraphlayer0.dummy(), ctx.x.scale_back1(ctx.r.gradp())

class SubgraphLayer0_normalize_channels(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=subgraphlayer0(1)
        r.obj=x.obj.normalize_columns()
        ctx.x=x.obj
        ctx.r=r.obj
        return r;

    @staticmethod
    def backward(ctx,g):
        ctx.x.normalize_columns_back(ctx.r)
        return subgraphlayer0.dummy()
    
class Subgraph_layer0_linearFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y,b):
        R=ptens.subgraphlayer0.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_linear(x.obj,y,b)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linear_back0(ctx.r.gradp(),ctx.y)
        return subgraphlayer0.dummy(), ctx.x.linear_back1(ctx.r.gradp()), ctx.x.linear_back2(ctx.r.gradp())



class SubgraphLayer0_ReLUFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,alpha):
        r=subgraph_layer(1)
        r.obj=_subgraphlayer0.zeros_like(x.obj)
        r.obj.add_ReLU(x.obj,alpha)
        ctx.x=x.obj
        ctx.alpha=alpha
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_ReLU_back(ctx.r,ctx.alpha)
        return subgraphlayer0.dummy(), None




# class Subgraph_layer0_Linmaps0Fn(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx,x):
#         R=ptens.subgraph_layer0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
#         ptens_base.add_linmaps0to0(R.obj,x.obj)
#         ctx.x=x.obj
#         ctx.r=R.obj
#         return R
        
#     @staticmethod
#     def backward(ctx,g):
#         ptens_base.add_linmaps0to0_back(ctx.x.gradp(),ctx.r.gradp())
#         return subgraph_layer0.dummy()


# class Subgraph_layer0_Linmaps1Fn(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx,x):
#         R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
#         ptens_base.add_linmaps0to1(R.obj,x.obj)
#         ctx.x=x.obj
#         ctx.r=R.obj
#         return R
        
#     @staticmethod
#     def backward(ctx,g):
#         ptens_base.add_linmaps0to1_back(ctx.x.gradp(),ctx.r.gradp())
#         return subgraph_layer0.dummy()


# class Subgraph_layer0_Linmaps2Fn(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx,x):
#         R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*2,x.obj.get_dev())
#         ptens_base.add_linmaps0to2(R.obj,x.obj)
#         ctx.x=x.obj
#         ctx.r=R.obj
#         return R
        
#     @staticmethod
#     def backward(ctx,g):
#         ptens_base.add_linmaps0to2_back(ctx.x.gradp(),ctx.r.gradp())
#         return subgraph_layer0.dummy()


class SubgraphLayer0_GatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,S):
        ctx.x=x
        r=ptens.subgraphlayer0(1)
        r.obj=_subgraph_layer0(x.obj,S.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        r.obj.add_gather_back(x.obj)
        return subgraphlayer0.dummy(), None, None


class Subgraph_layer0_Outer0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptens.subgraphlayer0(1)
        r.obj=ptens_base.outer(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return subgraphlayer0.dummy(), subgraphlayer0.dummy()


class Subgraph_layer0_Outer1Fn(torch.autograd.Function):

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
        return subgraphlayer0.dummy(), ptens.ptensors1.dummy()


class Subgraph_layer0_Outer2Fn(torch.autograd.Function):

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
        return subgraphlayer0.dummy(), ptens.ptensors2.dummy()






