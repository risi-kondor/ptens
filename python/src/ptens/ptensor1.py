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
    def from_matrix(self, _atoms, M):
        R=ptensor1(M)
        R.atoms=_atoms
        return R

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


    # ---- Operations ----------------------------------------------------------------------------------------


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "<ptensor1(atoms="+str(self.atoms)+",nc="+str(self.size(1))+")>"

    def __str__(self):
        return pb.ptensor1.view(self,self.atoms).str()
