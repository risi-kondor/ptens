.. ptens documentation master file, created by
   sphinx-quickstart on Sun Sep 11 21:20:11 2022.

ptens documentation
===================

`ptens` is a C++/CUDA library for permutation equivariant message passing and 
higher order equivariant graph neural networks. 
For an introduction to the Ptensors formalism, see this 
`preprint <https://arxiv.org/abs/2306.10767>`_. 
`ptens` can perform equivariant message between Ptensors on both the CPU and the GPU. 
Importantly, the library can perform this process in parallel across many pairs 
of source and destination Ptensors, even when the Ptensors are of different sizes and 
their reference domains overlap in different ways. 
This document describes `ptens`'s PyTorch interface.  

`ptens` is developed by Risi Kondor's research group and is released for noncommercial use 
under a custom license that can be found in the file LICENSE.TXT

|

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro.rst
   ptensork.rst
   ptensorsk.rst
   linmaps.rst
   transfer.rst
   graph.rst
   gather.rst 
   unite.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
