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

class ptensor(torch.Tensor):

    def __new__(cls, atoms:list, data:torch.Tensor | torch.Size, *args, **kwargs):
        # We write a new __new__ function here, since the signature now includes atoms.
        # But we need __new__ since it handles the memory allocations, potentially on the GPU.
        return torch.Tensor.__new__(cls, data, *args, **kwargs)

    def __init__(self, atoms:list, data:torch.Tensor | torch.Size, *args, **kwargs):
        """
        Instantiate a generic P-Tensor.
        It requires `atoms` and a tensor `data` structure to be passed to the tensor.
        See torch.Tensor for more details about the `data` argument and additional arguments supplied.
        """
        self.atoms = atoms

    @classmethod
    def make(cls, atoms:list, M:torch.Tensor | torch.Size):
        R = cls(atoms, data=M)
        return R

    # ---- Operations ----------------------------------------------------------------------------------------


    def __add__(self,y):
        assert self.size()==y.size()
        assert self.atoms==y.atoms
        return self.make(self.atoms,super().__add__(y))


    def __str__(self):
        return self.backend().str()

    def to_string(self,indent):
        return self.backend().str(indent)
