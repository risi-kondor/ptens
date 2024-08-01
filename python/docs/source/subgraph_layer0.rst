***************
Subgraph layers
***************

GNN applications involve Ptensor layers corresponding to different subgraphs. 
`ptens` provides specialized classes for this purpose derived from the generic 
`ptensorlayer0`, `ptensorlayer1` and `ptensorlayer2` classes. 
Instances of each of these `subgraphlahyer` classes must explicitly refer to the underlying 
graph `G`. 

========================
Defining subgraph layers
========================

The input layer of a GNN is typically a ``subgraphlayer0`` layuer in which each vertex has its own 
(zeroth order) Ptensor. Such a layers are easy to create with the ``from_matrix`` constructor 
without even having to specify its reference domains. 

.. code-block:: python

 >> G=ptens.ggraph.random(5,0.5)
 >> M=torch.randn(5,3)
 >> A=ptens.subgraph_layer0.from_matrix(G,M)
 >> print(A)

 Ptensor0 [0]:
   [ 0.157138 -0.0620347 0.859654 ]

 Ptensor0 [1]:
   [ 1.29978 -0.224464 -0.210561 ]

 Ptensor0 [2]:
   [ 2.0959 0.567439 -0.279718 ]

 Ptensor0 [3]:
   [ -0.360948 -0.495358 -0.238531 ]

 Ptensor0 [4]:
   [ -0.452405 -0.739714 0.266817 ]


Similarly to ``ptensorlayer`` classes, ``subgraphlayer`` classes also provides ``zero``, ``gaussian`` and 
``sequential`` constructors, but now we must also specify ``G``. 

.. code-block:: python

  >> A=ptens.subgraph_layer0.sequential(G,3)
  >> print(A)

  Ptensor0 [0]:
    [ 0 1 2 ]

  Ptensor0 [1]:
    [ 3 4 5 ]

  Ptensor0 [2]:
    [ 6 7 8 ]

  Ptensor0 [3]:
    [ 9 10 11 ]

  Ptensor0 [4]:
    [ 12 13 14 ]

For first and second order subgraph layers we must also specify what subgraph ``S`` the layer corresponds to. 
The reference domains of the Ptensors will correspond to all occurrences of ``S`` in ``G`` as described in 
the section on `finding subgraphs <subgraph.html#finding-subgraphs>`_.

.. code-block:: python

 >> G=ptens.ggraph.random(3,0.5)
 >> S=ptens.subgraph.triangle()
 >> A=ptens.subgraph_layer1.sequential(G,S,3)
 >> print(A)

 Ptensor1 [0,1,2]:
   [ 0 1 2 ]
   [ 3 4 5 ]
   [ 6 7 8 ]

 Ptensor1 [0,1,2]:
   [ 0 1 2 ]
   [ 3 4 5 ]
   [ 6 7 8 ]

 Ptensor1 [0,1,2]:
   [ 0 1 2 ]
   [ 3 4 5 ]
   [ 6 7 8 ]


===============
Message passing 
===============

The main advantage of subgraph layers is the ease with which they support message passing. 
The following code creates an input layer as before and then creates a first order layer corresponding 
to the edges in `G`. 
The `gather` operator ensures that each subgraph in the second layer collects equivariant messages 
from each subgraph in the first layer that it has any overlap with. Since in this case the 
"subgraphs" in the `f0` are just the vertices, effectively this realizes vertex-to-edge message passing. 

.. code-block:: python

 >> G=ptens.ggraph.random(5,0.5)
 >> M=torch.randn(5,3)
 >> f0=ptens.subgraph_layer0.from_matrix(G,M)

 >> S=ptens.subgraph.edge()
 >> f1=ptens.subgraph_layer1.gather(f0,S)

 >> print(f1)

 Ptensor1 [0,1]:
   [ -1.72529 2.43712 0.214614 ]
   [ -0.296102 -0.803141 -0.0876771 ]

 Ptensor1 [0,3]:
   [ -1.72529 2.43712 0.214614 ]
   [ 1.16169 0.409076 1.21103 ]

 Ptensor1 [1,2]:
   [ -0.296102 -0.803141 -0.0876771 ]
   [ -0.989146 -0.334836 0.65888 ]

The `gather` operator works similarly for message passing from a subgraph layer of any order to 
a subgraph layer of any order. 






