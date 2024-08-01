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


`Ptens` stores ``ggraph`` s in a custom sparse data structure, making it possible to 
handle graphs with a large number of vertices. 
The adjacency matrix can be extracted using the ``adjacency_matrix()``  method:

.. code-block:: python

 >> G.adjacency_matrix()

 tensor([[0., 1., 0., 1.],
         [1., 0., 1., 0.],
         [0., 1., 0., 0.],
         [1., 0., 0., 0.]])

=======
Caching 
=======

Graph neural network applications often involve learning from data on a large number of distinct graphs. 
For each graph, `ptens` needs to compute various objects such as the subgraph lists, layer maps, and so on. 
To reduce the burden of continually recomputing these objects, `ptens` makes it possible to cache the graphs, 
as well as most of the derived data structures. 

To add a given graph to `ptens` 's global graph cache, we simply need to assign it an id and call 
the ``cache`` method:

.. code-block:: python

 >> G1=ptens.ggraph.random(6,0.5)
 >> print(G1)
 >> G1.cache(3)

 Ggraph on 6 vertices:
   [ 0 1 0 1 0 1 ]
   [ 1 0 1 0 0 0 ]
   [ 0 1 0 0 0 0 ]
   [ 1 0 0 0 0 0 ]
   [ 0 0 0 0 0 1 ]
   [ 1 0 0 0 1 0 ]

The graph can then be retrieved at any later point using the ``from_cache`` constructor:
 
.. code-block:: python

 >> G2=ptens.from_cache(3)
 >> print(G2)

 Ggraph on 6 vertices:
   [ 0 1 0 1 0 1 ]
   [ 1 0 1 0 0 0 ]
   [ 0 1 0 0 0 0 ]
   [ 1 0 0 0 0 0 ]
   [ 0 0 0 0 0 1 ]
   [ 1 0 0 0 1 0 ]

The actual graph cache is an object called ``ptens_base.ggraph_cache``. 
We can check the number of cached graphs with its ``size`` method:

.. code-block:: python

 >> print(ptens_base..ggraph_cache.size())

 1



