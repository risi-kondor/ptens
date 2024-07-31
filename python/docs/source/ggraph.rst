*******
Ggraphs
*******

One of the primary applications of permutation equivariant neural nets are graph neural 
networks. The ``ptens.ggraph`` class provides the necessary functionality to build graph neural nets 
consisting of Ptensor layers.

A ``ggraph`` object can be initialized directly from its adjacency matrix, represented as an :math:`n \times n` 
dense matrix:

.. code-block:: python

 >>> A=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.float32)
 >>> G=ptens.ggraph.from_matrix(A)

Alternatively, it can be initialized from an "edge index", which is just a :math:`2\times E` integer 
tensor, listing all the edges:

.. code-block:: python

 >>> A=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.float32)
 >>> G=ptens.graph.from_edge_index(A)

Graphs are stored in a custom sparse data structure, allowing `ptens` to potentially handle graphs with a 
large number of vertices. Graphs are printed out by listing the edges incident 
on each vertex. Continuing the above example:

.. code-block:: python

 >>> G
 0<-((1,1))
 1<-((0,1)(2,1))
 2<-((1,1))
 
The dense representation of the adjacency matrix is recovered using the ``torch()``  method:

.. code-block:: python

 >>> G.torch()
 tensor([[0., 1., 0.],
         [1., 0., 1.],
         [0., 1., 0.]])

Random (undirected) graphs can be constructed using the ``random`` constructor, by providing 
the number of vertices and the probability of there being an edge beween any two vertices:

.. code-block:: python

 >>> G=ptens.graph.random(8,0.3)
 >>> G.torch()
 tensor([[0., 0., 1., 1., 0., 1., 1., 1.],
         [0., 0., 1., 0., 1., 0., 0., 0.],
         [1., 1., 0., 1., 0., 0., 0., 0.],
         [1., 0., 1., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 1.],
         [1., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0., 1.],
         [1., 0., 0., 0., 1., 0., 1., 0.]])


 
