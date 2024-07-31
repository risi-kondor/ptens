*********
Subgraphs
*********


In Ptensor-based graph neural networks, the individual Ptensors are typically attached to small subgraphs 
of the underlying graph. `ptens` provides a separate class called ``subgraph`` to define these 
subgraphs. 

`subgraph` objects are initialized and retrieved the same way as ``ggraph`` s:

.. code-block:: python

 >>> A=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.float32)
 >>> S=ptens.subgraph.from_matrix(A)

.. code-block:: python

 >>> A=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.float32)
 >>> S=ptens.subgraph.from_edge_index(A)

.. code-block:: python

 >>> S
 0<-((1,1))
 1<-((0,1)(2,1))
 2<-((1,1))
 
.. code-block:: python

 >>> S.torch()
 tensor([[0., 1., 0.],
         [1., 0., 1.],
         [0., 1., 0.]])

Internally however ``subgraph``s are represented somewhat differently than regular ``ggraph`` objects, 
in particular, once constructed, every ``subgraph`` object is cached by the library's backend 
for the remainder of the given `ptens` session. 

As a convenience `ptens` also defines some special subgraphs such as edges and triangles:

.. code-block:: python

 >>> S=ptens.subgraph.edge()
 >>> print(S)
 Subgraph on 2 vertices:
   [ 0 1 ]
   [ 1 0 ]


 >>> S=ptens.subgraph.triangle()
 >>> print(S)
 Subgraph on 3 vertices:
   [ 0 1 1 ]
   [ 1 0 1 ]
   [ 1 1 0 ]

 
