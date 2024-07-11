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
import torch.linalg

import ptens_base as pb
from ptens_base import subgraph as _subgraph


class subgraph_cache:

    @classmethod
    def subgraphs(self):
        for x in pb.subgraph_cache_subgraphs():
            print(x) 

    
