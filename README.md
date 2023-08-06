# ptens

_ptens_ is a C++/CUDA library with a PyTorch front end interface for permutation equivariant message passing and 
constructing higher order equivariant graph neural networks based on the Ptensors formalism (https://arxiv.org/abs/2306.10767). 
On the GPU, the library can parallelize the message passing process across 
many pairs of Ptensors, even when the Ptensors are of different sizes and 
their reference domains overlap in different ways. 

_ptens_ is developed by Risi Kondor's research group and is released for noncommercial use 
under the license provided in LICENSE.txt. 

Documentation for the library's Python/PyTorch API can be found at https://risi-kondor.github.io/ptens-doc/.
