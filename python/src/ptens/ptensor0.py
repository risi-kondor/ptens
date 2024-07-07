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
import ptens_base as pb
import ptens as p
from ptens.ptensor import ptensor


class ptensor0(ptensor):

    @classmethod
    def make(self,atoms,M):
        R=ptensor0(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self, _atoms, _nc, device='cpu'):
        return self.make(_atoms,torch.zeros([_nc],device=device))

    @classmethod
    def randn(self, _atoms, _nc, device='cpu'):
        return self.make(_atoms,torch.randn([_nc],device=device))

    @classmethod
    def from_tensor(self, _atoms, M):
        return self.make(_atoms,M)

    def backend(self):
        return pb.ptensor0.view(self.atoms,self)
    

    # ----- Access -------------------------------------------------------------------------------------------


    def getd(self):
        return 1
    
    def get_nc(self):
        return self.size(0)
    

    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        if isinstance(x,ptensor0):
            return x
        if isinstance(x,p.ptensor1):
            return x.reduce0()
        if isinstance(x,p.ptensor2):
            return x.reduce0()




    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "<ptensor0(atoms="+str(self.atoms)+",nc="+str(self.size(0))+")>"

    def __str__(self):
        return self.backend().str()
