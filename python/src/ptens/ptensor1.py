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


class ptensor1(ptensor):

    @classmethod
    def zeros(self,atoms,_nc,device='cpu'):
        return self.make(atoms,torch.zeros([len(atoms),_nc],device=device))

    @classmethod
    def randn(self,atoms,_nc,device='cpu'):
        return self.make(atoms,torch.randn([len(atoms),_nc],device=device))

    @classmethod
    def sequential(self,atoms,nc,device='cpu'):
        assert isinstance(nc,int)
        return self.make(atoms,torch.tensor([i for i in range (0,len(atoms)*nc)],
                                            dtype=torch.float,device=device).reshape(len(atoms),nc))

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


    # ---- Linmaps -------------------------------------------------------------------------------------------


    @classmethod
    def linmaps(self,x):
        nc=x.get_nc()
        if isinstance(x,p.ptensor0):
            return self.broadcast0(x)
        if isinstance(x,p.ptensor1):
            r=ptensor1.zeros(x.atoms,2*nc)
            r[:,0:nc]=self.broadcast0(x.reduce0())
            r[:,nc:2*nc]=x
            return r
        if isinstance(x,p.ptensor2):
            r=ptensor1.zeros(x.atoms,5*nc)
            r[:,0:2*nc]=self.broadcast0(x.reduce0())
            r[:,2*nc:5*nc]=x.reduce1()
            return r


    # ---- Reductions ---------------------------------------------------------------------------------------


    def reduce0(self):
        return p.ptensor0.make(self.atoms,self.sum(dim=0))


    # ---- Broadcasting ---------------------------------------------------------------------------------------


    @classmethod
    def broadcast0(self,x):
        return self.make(x.atoms,x.unsqueeze(0).expand(len(x.atoms),x.size(0)).contiguous())


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
