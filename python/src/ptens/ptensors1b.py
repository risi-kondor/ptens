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
from ptens_base import ptensors1b as _ptensors1b
from ptens.utility import device_id as device_id
from ptensorsb import * 

#import ptens.ptensor0
#import ptens.ptensors1
#import ptens.ptensors2 


class ptensors1b(torch.Tensor):

    @classmethod
    def dummy(self):
        R=ptensors1b(1)
        return R

    @classmethod
    def init(self,obj):
        R=ptensors1b(1)
        R.obj=obj
        return R
    
    @classmethod
    def from_matrix(self,M,atoms=None):
        return Ptensors1b_fromMxFn.apply(M,atoms)
            
    @classmethod
    def zeros(self, _atoms, _nc, device='cpu'):
        R=ptensors1b(1)
        R.obj=_ptensors1b.create(_atoms,_nc,0,device_id(device))
        return R

    @classmethod
    def randn(self, _atoms, _nc, device='cpu'):
        R=ptensors1b(1)
        R.obj=_ptensors1b.create(_atoms,_nc,4,device_id(device))
        return R

    @classmethod
    def sequential(self, _atoms, _nc, device='cpu'):
        R=ptensors1b(1)
        R.obj=_ptensors1b.create(_atoms,_nc,3,device_id(device))
        return R

    def randn_like(self):
        return ptensors1b.init(self.obj.randn_like())
    
    @classmethod
    def cat(self,*args):
        return Ptensors1_catFn.apply(self,*args)


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()

    def get_grad(self):
        R=ptensors1b(1)
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
        return Ptensors1b_toMxFn.apply(self)
        
    def to(self, device='cpu'):
        return Ptensors1b_toFn.apply(self,device)


    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        return Ptensorsb_addFn.apply(self,y)

    def cat_channels(self,y):
        return Ptensors1b_cat_channelsFn.apply(self,y)

    def outer(self,y):
         return Ptensors1b_outerFn.apply(self,y)

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

    def relu(self,alpha=0.1):
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


class Ptensors1b_fromMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms=None):
        R=ptensors1b(1)
        if atoms is None:
            R.obj=_ptensors1b(x)
        else:
            R.obj=_ptensors1b(atoms,x)
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        return ctx.r.get_grad().torch(), None



class Ptensors1b_catFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,dummy,*args):
        r=ptensors1b.dummy()
        ctx.args=[x.obj for x in args]
        r.obj=_ptensors1b.cat(ctx.args)
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        offs=0
        dummies=[]
        for x in ctx.args:
            x.add_cat_back(ctx.r,offs)
            offs=offs+x.dim(0)
            dummies.append(ptensors1b.dummy())
        return None, *dummies


class Ptensors1b_outerFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptensors1b.dummy()
        r.obj=x.obj.outer(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ctx.x.outer_back0(ctx.r,ctx.y)
        ctx.y.outer_back0(ctx.r,ctxxy)
        return ptensors1b.dummy(), ptensors1b.dummy()

