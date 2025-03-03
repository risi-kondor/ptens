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
import ptens as p 
import ptens_base as pb 


class cptensorlayer(torch.Tensor):

    covariant_functions=[torch.Tensor.to,torch.Tensor.add,torch.Tensor.sub,torch.relu]

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in cptensorlayer.covariant_functions:
            r= super().__torch_function__(func, types, args, kwargs)
            r.atoms=args[0].atoms
        else:
            r= super().__torch_function__(func, types, args, kwargs)
            if isinstance(r,torch.Tensor):
                r=torch.Tensor(r)
        return r

    

