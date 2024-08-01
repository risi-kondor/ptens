********
Batching
********

Fully taking advantage of the computational capacity of GPUs in neural networks typically requires 
combining training instances into so-called minibatches. This is equally true of equivarant message passing 
networks. However, due to the variety of cached helper objects involved in this case (``atomspack``\s, 
``layer_map``\s and ``gather_plan``s) batching cannot simply be accomplished by concatenating tensors. 
Instead, `ptens` provides specialized `batched_ptensorlayer` and `batched_subgraphlayer` classes. 

======================
Batches Ptensor layers
======================

A batched Ptensor layer can be constructed from ``N`` individual Ptensor layer objects. 
For example, the following constructs a batch consisting of 3 copies of the same first order layer 
``a``: 

.. code-block:: python

 >> subatoms=ptens_base.atomspack.from_list([[1,3,4],[2,5],[0,2]])
 >> a=ptens.ptensorlayer1.randn(subatoms,3)
 >> A=ptens.batched_ptensorlayer1.from_ptensorlayers([a,a,a])
 >> print(A)

 batched_ptensorlayer1(size=3,nc=3):
   Ptensorlayer1:
     Ptensor1 [1,3,4]:
       [ -1.59921 2.74342 1.07462 ]
       [ -2.39966 1.30962 -0.423231 ]
       [ -0.891532 0.0210365 0.546666 ]
     Ptensor1 [2,5]:
       [ 2.81015 1.2396 -1.18559 ]
       [ -1.10245 1.79717 1.55835 ]
     Ptensor1 [0,2]:
       [ -0.873466 1.0114 -0.286389 ]
       [ -0.753351 1.37246 0.488635 ]

   Ptensorlayer1:
     Ptensor1 [1,3,4]:
       [ -1.59921 2.74342 1.07462 ]
       [ -2.39966 1.30962 -0.423231 ]
       [ -0.891532 0.0210365 0.546666 ]
     Ptensor1 [2,5]:
       [ 2.81015 1.2396 -1.18559 ]
       [ -1.10245 1.79717 1.55835 ]
     Ptensor1 [0,2]:
       [ -0.873466 1.0114 -0.286389 ]
       [ -0.753351 1.37246 0.488635 ]

   Ptensorlayer1:
     Ptensor1 [1,3,4]:
       [ -1.59921 2.74342 1.07462 ]
       [ -2.39966 1.30962 -0.423231 ]
       [ -0.891532 0.0210365 0.546666 ]
     Ptensor1 [2,5]:
       [ 2.81015 1.2396 -1.18559 ]
       [ -1.10245 1.79717 1.55835 ]
     Ptensor1 [0,2]:
       [ -0.873466 1.0114 -0.286389 ]
       [ -0.753351 1.37246 0.488635 ]

Batched layers store their data in a single matrix consisting of the concatenation of the individual 
layers. Therefore most generic equivariant operations (linear maps, `relu`_s, etc.) can be applied 
to them just as if they were a single layer.

The syntax of ``linmaps`` and ``gather`` operations is also identical to the syntax of the same operations 
on individal layers. For example,

=======================
Batches subgraph layers
=======================

