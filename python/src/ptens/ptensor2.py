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
from ptens_base import ptensor0 as _ptensor0
from ptens_base import ptensor1 as _ptensor1
from ptens_base import ptensor2 as _ptensor2

import ptens.ptensor0 
import ptens.ptensor1

class ptensor2(torch.Tensor):

    @classmethod
    def zeros(self, _atoms, _nc, device='cpu'):
        R=ptensor2(torch.zeros(len(_atoms),len(_atoms),_nc, device))
        R.atoms=_atoms
        return R
    
    @classmethod
    def randn(self, _atoms, _nc, device='cpu'):
        R=ptensor2(torch.randn(len(_atoms),len(_atoms),_nc, device))
        R.atoms=_atoms
        return R

    @classmethod
    def sequential(self, _atoms, _nc, device='cpu'):
        R=ptensor2(_ptensor2.sequential(_atoms,_nc,ptens.device_id(device)).torch())
        R.atoms=_atoms
        return R


    # ---- Access --------------------------------------------------------------------------------------------


    def get_nc(self):
        return self.size(2)


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def linmaps0(self):
        return Ptensor2_Linmaps0Fn.apply(self);

    def linmaps1(self):
        return Ptensor2_Linmaps1Fn.apply(self);

    def linmaps2(self):
        return Ptensor2_Linmaps2Fn.apply(self);


    def transfer0(self,_atoms):
        return Ptensor2_Transfer0Fn.apply(self,_atoms)

    def transfer1(self,_atoms):
        return Ptensor2_Transfer1Fn.apply(self,_atoms)

    def transfer2(self,_atoms):
        return Ptensor2_Transfer2Fn.apply(self,_atoms)


    # ---- I/O -----------------------------------------------------------------------------------------------


    def __str__(self):
        u=_ptensor2.view(self,self.atoms)
        return u.__str__()

    def __repr__(self):
        u=_ptensor2.view(self,self.atoms)
        return u.__str__()



# ------------------------------------------------------------------------------------------------------------


class Ptensor2_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensor0.zeros(x.atoms,2*x.get_nc())
        u=_ptensor2.view(x,x.atoms)
        r=_ptensor0.view(R,R.atoms)
        ptens_base.add_linmaps2to0(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor2.zeros(g.atoms,g.get_nc()/2)
        u=_ptensor0.view(g,g.atoms)
        r=_ptensor2.view(R,R.atoms)
        ptens_base.add_linmaps2to0_back(r,u)
        return R


class Ptensor2_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensor1.zeros(x.atoms,5*x.get_nc())
        u=_ptensor2.view(x,x.atoms)
        r=_ptensor1.view(R,R.atoms)
        ptens_base.add_linmaps2to1(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor2.zeros(g.atoms,g.get_nc()/5)
        u=_ptensor1.view(g,g.atoms)
        r=_ptensor2.view(R,R.atoms)
        ptens_base.add_linmaps2to1_back(r,u)
        return R


class Ptensor2_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptensor2.zeros(x.atoms,15*x.get_nc())
        u=_ptensor2.view(x,x.atoms)
        r=_ptensor2.view(R,R.atoms)
        ptens_base.add_linmaps2to2(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor2.zeros(g.atoms,int(g.get_nc()/15)) # !!
        u=_ptensor2.view(g,g.atoms)
        r=_ptensor2.view(R,R.atoms)
        ptens_base.add_linmaps2to2_back(r,u)
        return R


class Ptensor2_Transfer0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,_atoms):
        R=ptens.ptensor0.zeros(_atoms,x.get_nc()*2)
        u=_ptensor2.view(x,x.atoms)
        r=_ptensor0.view(R,R.atoms) 
        ptens_base.add_msg(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor2.zeros(g.atoms,g.get_nc()/2)
        u=_ptensor0.view(g,g.atoms)
        r=_ptensor2.view(R,R.atoms) 
        ptens_base.add_msg_back(r,u) 
        return R,None


class Ptensor2_Transfer1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,_atoms):
        R=ptens.ptensor1.zeros(_atoms,5*x.get_nc())
        u=_ptensor2.view(x,x.atoms)
        r=_ptensor1.view(R,R.atoms) 
        ptens_base.add_msg(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor2.zeros(g.atoms,g.get_nc()/5)
        u=_ptensor1.view(g,g.atoms)
        r=_ptensor2.view(R,R.atoms) 
        ptens_base.add_msg_back(r,u) 
        return R,None


class Ptensor2_Transfer2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,_atoms):
        R=ptensor2.zeros(_atoms,x.get_nc()*15)
        u=_ptensor2.view(x,x.atoms)
        r=_ptensor2.view(R,R.atoms) 
        ptens_base.add_msg(r,u)
        return R
        
    @staticmethod
    def backward(ctx,g):
        R=ptensor2.zeros(g.atoms,g.get_nc()/15)
        u=_ptensor2.view(g,g.atoms)
        r=_ptensor2.view(R,R.atoms) 
        ptens_base.add_msg_back(r,u) 
        return R,None



