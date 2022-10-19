******
Graphs
******

One of the primary applications of permutation equivariant neural nets is the contructon of graph neural 
networks. The ``ptens.graph`` class provides the necessary functionality to build graph neural nets 
consisting of Ptensor layers.

A ``graph`` object can be initialized directly from its adjacency matrix, provided as an :math:`n \times n` 
dense matrix:

.. code-block:: python

 >>> A=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.float32)
 >>> G=ptens.graph.from_matrix(A)

Graphs are stored in a custom sparse datastructure, allowing `ptens` to potentially handle graphs with a 
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
One feature of such networks is that thee references domains of the neurons correspond to 
the k-neighborhoods around each vertex in the graph. 
The ``nhoods(k)`` function returns these reference domains as ``atomspack`` objects. 

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


