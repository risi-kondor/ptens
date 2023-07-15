******
Graphs
******

One of the primary applications of permutation equivariant neural nets are graph neural 
networks. The ``ptens.graph`` class provides the necessary functionality to build graph neural nets 
consisting of Ptensor layers.

A ``graph`` object can be initialized directly from its adjacency matrix, represented as an :math:`n \times n` 
dense matrix:

.. code-block:: python

 >>> A=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.float32)
 >>> G=ptens.graph.from_matrix(A)

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

=============
Neighborhoods
=============

The ``graph`` class provides some specialized functionality required by equivariant graph neural nets. 
In certain classes of GNNs, the references domains of the neurons correspond to 
the k-neighborhoods around each vertex in the graph. 
The ``nhoods(k)`` function returns these reference domains as an ``atomspack`` object. 

.. code-block:: python

 >>> G=ptens.graph.random(8,0.2)
 >>> G.nhoods(0)
 (0)
 (1)
 (2)
 (3)
 (4)
 (5)
 (6)
 (7)

 >>> G.nhoods(1)
 (0,5)
 (1,4,7)
 (2,5)
 (3)
 (1,4)
 (0,2,5,6)
 (5,6)
 (1,7)

 >>> G.nhoods(2)
 (0,2,5,6)
 (1,4,7)
 (0,2,5,6)
 (3)
 (1,4,7)
 (0,2,5,6)
 (0,2,5,6)
 (1,4,7)


===================
Edges and subgraphs
===================

The ``edges()`` method returns the list of edges in ``G``:

.. code-block:: python

 >>> G=ptens.graph.random(8,0.3)
 >>> E=G.edges()
 >>> print(E)
 (0,1)
 (1,0)
 (1,2)
 (2,1)
 (2,7)
 (4,5)
 (4,6)
 (4,7) 
 (5,4)
 (6,4)
 (7,2)
 (7,4)

More generally, if we define a second graph ``H``, the ``subgraphs(H)`` method finds all occurrences of ``H`` 
in ``G`` as a subgraph and returns the result as an ``atomspack``:

.. code-block:: python

 >>> G=ptens.graph.random(8,0.6)
 >>> triangle=ptens.graph.from_matrix(torch.tensor([[0,1,1],[1,0,1],[1,1,0]],dtype=torch.float32))
 >>> S=G.subgraphs(triangle)
 >>> S
 (0,1,6)
 (0,3,6)
 (0,5,6)
 (1,4,6)
 (4,5,6)
 (4,5,7)
 (4,6,7)
 (5,6,7)

========
Overlaps
========

Given two ``atomspack`` objects, the ``overlaps(A,B)`` method creates a bipartite graph in which 
there is an edge from ``i`` to ``j`` if the ``i`` 'th set in ``A`` has a non-zero intersection with the 
``j`` 'th set in ``B``. 

.. code-block:: python

 >>> A=ptens_base.atomspack([[0,1],[2],[4,5]])
 >>> B=ptens_base.atomspack([[1,3],[5,2],[0]])
 >>> G=ptens.graph.overlaps(A,B)
 >>> G.torch()
 tensor([[1., 0., 1.],
         [0., 1., 0.],
         [0., 1., 0.]])

 
