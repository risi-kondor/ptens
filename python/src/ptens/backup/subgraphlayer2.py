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
from ptens_base import subgraph_layer2 as _subgraph_layer2
from ptens.utility import device_id as device_id

import ptens.ptensors0 
import ptens.ptensors1 
import ptens.ptensors2 

from ptens.subgraphlayer import *


class subgraphlayer2(subgraphlayer):

    @classmethod
    def dummy(self):
        R=subgraphlayer2(1)
        return R

    @classmethod
    def raw(self,G,S,_atoms,_nc,device=0):
        R=subgraphlayer2(1)
        R.obj=_subgraph_layer2.raw(G.obj,S.obj,_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def zeros(self,G,S,_atoms,_nc,device=0):
        R=subgraphlayer2(1)
        R.obj=_subgraph_layer2.zero(G.obj,S.obj,_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def randn(self,G,S,_atoms,_nc,device=0):
        R=subgraphlayer2(1)
        R.obj=_subgraph_layer2.gaussian(G.obj,S.obj,_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def sequential(self,G,S,_atoms,_nc,device=0):
        R=subgraphlayer2(1)
        R.obj=_subgraph_layer2.sequential(G.obj,S.obj,_atoms,_nc,ptens.device_id(device))
        return R

    @classmethod
    def gather(self,x,S):
        return SubgraphLayer2_GatherFn.apply(x,S)
    
    @classmethod
    def gather(self,x,G,S):
        return SubgraphLayer2_GatherFromPtensorsFn.apply(x,G,S)
    
    
    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=subgraphlayer2(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=subgraphlayer2(1)
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

    #def __getitem__(self,i):
    #    return Subgraph_layer2_getFn.apply(self,i)
    
    def push_back(self, x):
        return self.obj.push_back(x)

    def randn_like(self,sigma=1.0):
        return subgraphlayer2.randn(self.get_atoms(),self.get_nc(),sigma,self.get_dev())

    def torch(self):
        return SubgraphLayer2_toMxFn.apply(self)

    def to(self, device='cpu'):
        return SubgraphLayer_toDeviceFn.apply(self,device)


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def __add__(self,y):
        return SubgraphLayer_plusFn.apply(self,y)

    def __mul__(self,y):
        return SubgraphLayer_mprodFn.apply(self,y)

    def linear(self,y,b):
        return SubgraphLayer_linearFn.apply(self,y,b)

    def concat(self,y):
        return SubgraphLayer_concatFn.apply(self,y)
    
    def relu(self,alpha=0.5):
        return SubgraphLayer2_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return SubgraphLayer_inpFn.apply(self,y)
    
    def diff2(self,y):
        return ptens.SubgraphLayer2_diff2Fn.apply(self,y)

    def outer(self,y):
        if isinstance(y,ptens.ptensors0):
            return Subgraph_layer2_Outer0Fn.apply(self,y)
        if isinstance(y,ptens.subgraphlayer2):
            return Subgraph_layer2_Outer1Fn.apply(self,y)

    def scale(self,y):
        return Subgraph_layer2_scaleFn.apply(self,y)

    def mult_channels(self,y):
        return Subgraph_layer2_mult_channelsFn.apply(self,y)


    # ---- Message passing -----------------------------------------------------------------------------------


#     def linmaps0(self,normalized=False):
#         return Subgraph_layer2_Linmaps0Fn.apply(self,normalized);
#     def linmaps1(self,normalized=False):
#         return Subgraph_layer2_Linmaps1Fn.apply(self,normalized);
#     def linmaps2(self,normalized=False):
#         return Subgraph_layer2_Linmaps2Fn.apply(self,normalized);


    def transfer0(self,_atoms,G,normalized=False):
        return Subgraph_layer2_Transfer0Fn.apply(self,_atoms,G,normalized)

    def transfer1(self,_atoms,G,normalized=False):
        return Subgraph_layer2_Transfer1Fn.apply(self,_atoms,G,normalized)

    def transfer2(self,_atoms,G,normalized=False):
        return Subgraph_layer2_Transfer2Fn.apply(self,_atoms,G,normalized)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------

    
class SubgraphLayer2_fromMxFn(torch.autograd.Function): #TODO 
    @staticmethod
    def forward(ctx,G,x):
        R=subgraphlayer1(1)
        R.obj=_subgraph_layer2(G.obj,x)
        return R
    @staticmethod
    def backward(ctx,g):
        return None, ctx.r.get_grad().torch()


class SubgraphLayer2_toMxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.x=x.obj
        return x.obj.torch()
    @staticmethod
    def backward(ctx,g):
       R=subgraphlayer2(1)
       ctx.x.torch_back(g)
       return R
    


# class Subgraph_layer2_getFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,i):
#         R=ptens.ptensor1(x.obj[i].torch())
#         R.atoms=x.atoms_of(i)
#         ctx.x=x.obj
#         ctx.i=i
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         R=subgraphlayer2(1)
#         ctx.x.add_to_grad(ctx.i,g)
#         return R,None


# class Subgraph_layer2_toFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,_dev):
#         dev=ptens.device_id(_dev)
#         R=subgraphlayer2(1)
#         R.obj=_subgraph_layer2(x.obj,dev)
#         ctx.x=x.obj
#         ctx.r=R.obj
#         ctx.dev=dev
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.move_to_device_back(ctx.r.get_gradp(),ctx.dev)
#         return subgraphlayer2.dummy(), None
        
    
# class SubgraphLayer2_addFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         R=subgraphlayer2(1)
#         R.obj=_subgraph_layer2(x.obj)
#         R.obj.add(y.obj)
#         ctx.x=x.obj
#         ctx.y=y.obj
#         ctx.r=R.obj
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_to_grad(ctx.r.get_gradp())
#         ctx.y.add_to_grad(ctx.r.get_gradp())
#         return subgraphlayer2(1),subgraphlayer2(1)


# class Subgraph_layer2_ReLUFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,alpha):
#         R=ptens.subgraphlayer2.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
#         R.obj.add_ReLU(x.obj,alpha)
#         ctx.x=x.obj
#         ctx.alpha=alpha
#         ctx.r=R.obj
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_ReLU_back(ctx.r,ctx.alpha)
#         return subgraphlayer2.dummy(), None


# class Subgraph_layer2_inpFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         ctx.x=x.obj
#         ctx.y=y.obj
#         return torch.tensor(x.obj.inp(y.obj))
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_to_grad(ctx.y,g.item())
#         ctx.y.add_to_grad(ctx.x,g.item())
#         return subgraphlayer2.dummy(), subgraphlayer2.dummy()


# class Ptensors0_diff2Fn(torch.autograd.Function):
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
#         return subgraph_layer2.dummy(), subgraph_layer2.dummy()


# class Subgraph_layer2_concatFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         r=subgraphlayer2(1)
#         r.obj=_subgraph_layer2.concat(x.obj,y.obj)
#         ctx.x=x.obj
#         ctx.y=y.obj
#         ctx.r=r.obj
#         return r
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_concat_back(ctx.r,0)
#         ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
#         return subgraphlayer2(1),subgraphlayer2(1)


# class Subgraph_layer2_mprodFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y):
#         R=ptens.subgraphlayer2.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
#         R.obj.add_mprod(x.obj,y)
#         ctx.x=x.obj
#         ctx.y=y
#         ctx.r=R.obj
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
#         return subgraphlayer2(1), ctx.x.mprod_back1(ctx.r.gradp())


# class Subgraph_layer2_linearFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,y,b):
#         R=ptens.subgraphlayer2.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
#         R.obj.add_linear(x.obj,y,b)
#         ctx.x=x.obj
#         ctx.y=y
#         ctx.r=R.obj
#         return R
#     @staticmethod
#     def backward(ctx,g):
#         ctx.x.add_linear_back0(ctx.r.gradp(),ctx.y)
#         return subgraph_layer2.dummy(), ctx.x.linear_back1(ctx.r.gradp()), ctx.x.linear_back2(ctx.r.gradp())


class Subgraph_layer2_scaleFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_back0(ctx.r.gradp(),ctx.y)
        return ptensors0.dummy(), ctx.x.scale_back1(ctx.r.gradp())


class Subgraph_layer2_mult_channelsFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.subgraphlayer2.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_scale_channels(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_scale_channels_back0(ctx.r.gradp(),ctx.y)
        return subgraphlayer2.dummy(), None



class SubgraphLayer2_GatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,S):
        ctx.x=x
        r=ptens.subgraphlayer2(1)
        r.obj=_subgraph_layer2(x.obj,S.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.r.gather_back(ctx.x)
        return subgraphlayer2.dummy(), None, None


class SubgraphLayer2_GatherFromPtensorsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,G,S):
        ctx.x=x
        r=ptens.subgraphlayer2(1)
        r.obj=_subgraph_layer2(x.obj,G.obj,S.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        return r
    @staticmethod
    def backward(ctx,g):
        ctx.r.gather_back(ctx.x)
        return subgraphlayer2.dummy(), None, None, None




