******
Graphs
******

One of the primary applications of P-tensors are graph neural 
networks. The underlying graph must be stored in a ``ptens.ggraph`` object. 

A ``ggraph`` can be constructed directly from its adjacency matrix, represented as a dense :math:`n \times n` 
matrix:

.. code-block:: python

 >> A=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.float32)
 >> G=ptens.ggraph.from_matrix(A)

 Ggraph on 3 vertices:
   [ 0 1 0 ]
   [ 1 0 1 ]
   [ 0 1 0 ]

Alternatively, it can be initialized from an "edge index", which is a :math:`2\times E` integer 
tensor, listing all the edges:

.. code-block:: python

 >> A=torch.tensor([[0,1,1,2,0,3],[1,0,2,1,3,0]],dtype=torch.float32)
 >> G=ptens.ggraph.from_edge_index(A)

 Ggraph on 4 vertices:
   [ 0 1 0 1 ]
   [ 1 0 1 0 ]
   [ 0 1 0 0 ]
   [ 1 0 0 0 ]

Random (undirected) graphs can be constructed using the ``random`` constructor, providing 
the number of vertices and the probability of there being an edge beween any two vertices:

.. code-block:: python

  >> G=ptens.ggraph.random(8,0.3)
  >> G.torch()

  tensor([[0., 0., 1., 1., 0., 1., 1., 1.],
         [0., 0., 1., 0., 1., 0., 0., 0.],
	  [1., 1., 0., 1., 0., 0., 0., 0.],
	  [1., 0., 1., 0., 0., 0., 0., 0.],
	  [0., 1., 0., 0., 0., 0., 0., 1.],
	  [1., 0., 0., 0., 0., 0., 0., 0.],
	  [1., 0., 0., 0., 0., 0., 0., 1.],
	  [1., 0., 0., 0., 1., 0., 1., 0.]])


``ggraph`` s are stored in a custom sparse data structure, allowing `ptens` to handle graphs with  
a large number of vertices. The adjacency matrix can be recovered in dense format using the 
``torch()``  method:

.. code-block:: python

 >> G.torch()

 tensor([[0., 1., 0., 1.],
         [1., 0., 1., 0.],
         [0., 1., 0., 0.],
         [1., 0., 0., 0.]])

=======
Caching 
=======