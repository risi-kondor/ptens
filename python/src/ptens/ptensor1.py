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
from ptens.ptensor import ptensor


class ptensor1(ptensor):

    @classmethod
    def make(self,atoms,M):
        R=ptensor1(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,_nc,device='cpu'):
        R=ptensor1(torch.zeros([len(atoms),_nc],device=device))
        R.atoms=atoms
        return R

    @classmethod
    def randn(self,atoms,_nc,device='cpu'):
        R=ptensor1(torch.randn([len(atoms),_nc],device=device))
        R.atoms=atoms
        return R

    @classmethod
    def from_tensor(self, _atoms, M):
        return self.make(_atoms,M)

    def backend(self):
        return pb.ptensor1.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getd(self):
        return self.size(0)
    
    def get_nc(self):
        return self.size(1)
    

    # ---- Reductions ---------------------------------------------------------------------------------------


    def reduce0(self):
        return self.sum(dim=0)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "<ptensor1(atoms="+str(self.atoms)+",nc="+str(self.size(1))+")>"

    def __str__(self):
        return self.backend().str()






#    def clone(self):
#        r=ptensor1(super().clone())
#        r.atoms=self.atoms
#        return r

#     def __copy__(self):
#         print("copied")
#         return self.__class__(self.clone())

#     def __deepcopy__(self, memo):
#         print("deep copied")
#         return self.__class__(copy.deepcopy(self.data, memo))


