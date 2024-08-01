********
Batching
********

Fully taking advantage of the computational capacity of GPUs in neural networks usually requires 
combining training instances into so-called minibatches. This is equally true of equivarant message passing 
networks. However, due to the variety of cached helper objects involved in this case (``atomspack``\s, 
``layer_map``\s and ``gather_plan``\s) batching cannot simply be accomplished by concatenating tensors. 
Instead, `ptens` provides the specialized classes ``batched_ggraph``, ``batched_atomspack``, 
``batched_ptensorlayer``, ``batched_subgraphlayer`` and so on for this purpose. 

======================
Batched Ptensor layers
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

Alternatively, it can be initilized from a ``batched_atomspack`` object using the usual ``zeros``, 
``randn`` or ``sequential`` constructors or from a large matrix that holds the concatenation of 
each of the layers:

.. code-block:: python

 >> atoms=ptens_base.batched_atomspack([subatoms,subatoms,subatoms])
 >> M=torch.randn([atoms.nrows1(),3])
 >> A=ptens.batched_ptensorlayer1.from_matrix(atoms,M)

Batched layers store their data in a single matrix consisting of the concatenation of the individual 
layers. Therefore, many generic operations (linear maps, ``relu``\s, etc.) can be applied 
to them just as if they were a single layer.

In contrast, to avoid excessive coping, unlike in some other packages such as Pytorch Geometric, 
the batched control data structures such as ``batched_atomspack`` and ``batched_atomspack``, etc., 
just hold pointers to the underlying objects rather than explicitly concatenating them. 
However, the interface of these classes exactly mirrors the interface of their non-batched versions. 
For example, the following code creates the overlap maps between two batches of ``atomspack``\s:

.. code-block:: python

 >> a1=ptens_base.atomspack.random(3,6,0.5)
 >> a2=ptens_base.atomspack.random(4,6,0.5)
 >> batched_atoms1=ptens_base.batched_atomspack([a1,a1,a1])
 >> batched_atoms2=ptens_base.batched_atomspack([a2,a2,a2])
 >> L=ptens_base.batched_layer_map.overlaps_map(batched_atoms2,batched_atoms1)

The syntax of the ``linmaps`` and ``gather`` operations is also the same as for individual layers:

.. code-block:: python

 >> A=ptens.batched_ptensorlayer1.randn(batched_atoms1,3)
 >> B=ptens.batched_ptensorlayer1.gather(batched_atoms2,A) 

The performance gains of batching come largely from the fact that on the backend `ptens` can perform these 
operations in single CUDA kernel call. 

=======================
Batched subgraph layers
=======================

Just like batched atomspacks, batched ``ggraph`` instances are easy to create:

.. code-block:: python

 >> G0=ptens.ggraph.random(6,0.5)
 >> G1=ptens.ggraph.random(6,0.5)
 >> G2=ptens.ggraph.random(6,0.5)
 >> G=ptens.batched_ggraph.from_graphs([G0,G1,G2])

We can then create a ``batched_subgraphlayer`` for a given subgraph ``S`` using one of the usual constructors: 

.. code-block:: python

 >> A=p.batched_subgraphlayer1.randn(G,S,3)

or from a PyTorch matrix: 

.. code-block:: python

 >> M=torch.randn([G.subgraphs(S).nrows1(),3])
 >> A=p.batched_subgraphlayer1.from_matrix(G,S,M)

Note that the `subgraph` object is not batched and all the subgraphlayers in the batch must correspond to 
the `same` subgraph. The ``linmaps`` and ``gather`` operations generalize to batched subgraphlayers as 
expected, for example we can write:

 >> T=ptens.subgraph.triangle()
 >> B1=ptens.batched_subgraphlayer1.gather(T,A)

Just as for the ``batched_ptensorlayer`` classes, `ptens` executes these operations on GPUs in a 
highly optimized way, parallelizing over both members of the batch and the individual Ptensors inside them. 


