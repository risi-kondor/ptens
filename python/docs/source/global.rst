*******************************
Global parameters and variables
*******************************

==================
GPU memory manager
==================

Certain `ptens` operations, specifically ``linmaps`` and ``gather``, involve creating large 
temporary objects.  
However, memory allocation and deallocation on GPUs can have a large latency. 
To remedy this problem, the `ptens` backend can use a fast custom memory manager. 
The amount of GPU memory allocated to this memory manager (in MB) 
can be set with the `vram_manager.reset` function:

.. code-block:: python

 >> vram_manager.reset(1000)

This variable should be set with care to find the right balance between the fraction of 
memory available to PyTorch objects vs. the `ptens` backend.

============
ptens status
============

The ``ptens_base.status_str`` function returns information about the current `ptens` session in a 
string:
 
.. code-block:: python

 >> print(ptens_base.status_str())

 ---------------------------------------
  Ptens 0.0 
  CUDA support:                     OFF
 
  AtomsPack cat cache:                0
  Graph cache:                        0
  Graph elist cache:                  0
  Subgraph cache:                     0
 ---------------------------------------



