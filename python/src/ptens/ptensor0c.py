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
import ptens.ptensorc_base as ptensorc_base


class ptensor0c(ptensorc_base):

    @classmethod
    def zeros(self, _atoms, _nc, device='cpu'):
        R=ptensor0c(torch.zeros([_nc],device=device))
        R.atoms=_atoms
        return R

    @classmethod
    def randn(self, _atoms, _nc, device='cpu'):
        R=ptensor0c(torch.randn([_nc],device=device))
        R.atoms=_atoms
        return R

    @classmethod
    def from_matrix(self, _atoms, M):
        R=ptensor0c(M)
        R.atoms=_atoms
        return R

    def clone(self):
        r=ptensor0c(super().clone())
        r.atoms=self.atoms
        return r


    # ---- Operations ----------------------------------------------------------------------------------------


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "ptensor0c("+str(self.atoms)+","+str(self.size(0))+")"

    def __str__(self):
        r="Ptensor0("+str(self.atoms)+"):\n"
        r=r+torch.Tensor(self).__str__()
        return r