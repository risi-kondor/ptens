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
from ptens.utility import device_id as device_id

import ptens.ptensor0
#import ptens.ptensors1
#import ptens.ptensors2 

from ptens.ptensorsb import *


class ptensors0b(ptensorsb):

    @classmethod
    def dummy(self):
        R=ptensors0b(1)
        return R

    @classmethod
    def from_matrix(self,M,atoms=None):
        return Ptensors0b_fromMxFn.apply(M,atoms)
            
    @classmethod
    def zeros(self, _atoms, _nc, device='cpu'):
        R=ptensors0b(1)
        R.obj=_ptensors0b.create(_atoms,_nc,0,device_id(device))
        return R

    @classmethod
    def randn(self, _atoms, _nc, device='cpu'):
        R=ptensors0b(1)
        R.obj=_ptensors0b.create(_atoms,_nc,4,device_id(device))
        return R

    @classmethod
    def sequential(self, _atoms, _nc, device='cpu'):
        R=ptensors0b(1)
        R.obj=_ptensors0b.create(_atoms,_nc,3,device_id(device))
        return R

    def randn_like(self):
        return ptensors0b.randn(self.get_atoms(),self.get_nc(),self.get_dev())


    # ----- Access -------------------------------------------------------------------------------------------


    def __getitem__(self,i):
        return Ptensors0b_getFn.apply(self,i)
    


# ------------------------------------------------------------------------------------------------------------


class Ptensors0b_fromMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms):
        R=ptensors0b(1)
        if atoms:
            R.obj=_ptensors0b(x,atoms)
        else:
            R.obj=_ptensors0b(x)
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        return ctx.r.get_grad().torch(), None


class Ptensors0b_getFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,i):
        R=ptens.ptensor0(x.obj[i].torch())
        R.atoms=x.atoms_of(i)
        ctx.x=x.obj
        ctx.i=i
        return R

    @staticmethod
    def backward(ctx,g):
        R=ptensors0b(1)
        ctx.x.add_to_grad(ctx.i,g)
        return R,None




class Ptensors0b_GatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G):
        r=ptens.ptensors0b(1)
        r.obj=ptens_base.gather(x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.G=G.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.gather_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors0b.dummy(), None

