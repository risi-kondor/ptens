*********
Subgraphs
*********

In GNN applications, Ptensors are often associated to subgraphs of the underlying graph ``G``. 
`ptens` provides a separate class for defining these subgraphs. 

Some simple categories of subgraphs are predefined:

.. code-block:: python

  >> E=ptens.subgraph.edge()
  >> print(E)
  >> T=ptens.subgraph.triangle()
  >> print(T)
  >> C=ptens.subgraph.cycle(5)
  >> print(C)
  >> S=ptens.subgraph.star(5)
  >> print(S)

  Subgraph on 2 vertices:
    [ 0 1 ]
    [ 1 0 ]

  Subgraph on 3 vertices:
    [ 0 1 1 ]
    [ 1 0 1 ]
    [ 1 1 0 ]

  Subgraph on 5 vertices:
    [ 0 1 0 0 1 ]
    [ 1 0 1 0 0 ]
    [ 0 1 0 1 0 ]
    [ 0 0 1 0 1 ]
    [ 1 0 0 1 0 ]

  Subgraph on 5 vertices:
    [ 0 1 1 1 1 ]
    [ 1 0 0 0 0 ]
    [ 1 0 0 0 0 ]
    [ 1 0 0 0 0 ]
    [ 1 0 0 0 0 ]

Similarly to ``ggraph``\s, ``subgraph``\s can also be defined from their adjacency matrix: 

.. code-block:: python

 >> M=torch.tensor([[0,1,1],[1,0,1],[1,1,0]],dtype=torch.float)
 >> S=ptens.subgraph.from_matrix(M)
 >> print(S)

 Subgraph on 3 vertices:
   [ 0 1 1 ]
   [ 1 0 1 ]
   [ 1 1 0 ]

or from an edge list matrix: 

.. code-block:: python

 >> ix=torch.tensor([[0,1,2,0,3],[1,2,0,3,0]],dtype=torch.int)
 >> S=ptens.subgraph.from_edge_index(ix)
 >> print(S)

 Subgraph on 4 vertices:
   [ 0 1 1 1 ]
   [ 1 0 1 0 ]
   [ 1 1 0 0 ]
   [ 1 0 0 0 ]

=================
Finding subgraphs
=================

The primary purpose of defining a ``subgaph`` object ``S`` is to find all occurrences of ``S`` 
in a graph ``G`` (or a collection of graphs).  This is done with the ``ggraph`` class's 
``subgraph`` method: 

.. code-block:: python

 >> G=ptens.ggraph.random(8,0.5)
 >> S=ptens.subgraph.triangle()
 >> atoms=G.subgraphs(S)
 >> print(atoms)

 ([0,5,6],[4,7,6],[4,6,5])

The resulting ``atomspack`` object can be directly used to define a corresponding ``ptensorlayer``:

.. code-block:: python

 >> A=ptens.ptensorlayer1.randn(atoms,3)
 >> print(A)

 Ptensorlayer1:
   Ptensor1 [0,5,6]:
     [ 0.960329 -1.63022 0.106229 ]
     [ 0.884231 -0.0636849 -1.08168 ]
     [ 1.23821 0.29263 -1.1062 ]
   Ptensor1 [4,7,6]:
     [ -0.0967667 1.12721 -0.332577 ]
     [ -1.40149 1.47884 -1.15777 ]
     [ -0.446256 -1.18378 0.815759 ]
   Ptensor1 [4,6,5]:
     [ 1.00193 -2.19192 1.63382 ]
     [ 0.507325 -0.290758 -1.33027 ]
     [ -0.349507 -1.41685 -0.111342 ]

Finding subgraphs is a relatively expensive computation that has to be performed on the CPU. 
Therefore, the result of the operation is automatically cached, i.e., as long as the backend objects of 
``G`` and ``S`` are in scope, if the subgraphs isomorphic to ``S`` in ``G`` need to be found again, `ptens` 
will return the cached result. We can inspect all cached subgraph lists associated with a given ``G``:  

.. code-block:: python

 >> C=G.cached_subgraph_lists()
 >> print(C)
  {Subgraph on 3 vertices:
    [ 0 1 1 ]
    [ 1 0 1 ]
    [ 1 1 0 ]
    : ([0,5,6],[4,7,6],[4,6,5])}

One of the purposes of saving ``ggraph`` s in a cache (see `previous section <graph.html#caching>`_)  
is to ensure that they remain in scope, and consequently 
all the subgraph lists that have been computed for them also remain cached for future use. 

=======
Caching
=======

Typical GNN applications only involve a relatively small number of distinct subgraphs. 
Therefore, by default, `ptens` automatically caches the backend data structures corresponding to ``subgraph`` 
objects for the entirety of the library's run time time. 

For example, if a subgraph ``S1`` is defined from its adjacency matrix, 
and at some later point a second subgraph ``S2`` is defined with the same adjacency matrix, 
then `ptens` will make sure that ``S1`` and ``S2`` will point to the same underlying backend object. 
This makes it possible to reuse a variety of information related to ``S1``, 
including the related subgraph lists, layer maps and gather plans. 

The subgraph cache can be accessed via the ``ptens_base.subgraph_cache`` class:

.. code-block:: python

 >> C=pb.subgraph_cache.torch()
 >> for s in C:
      print(s)

 Subgraph on 4 vertices:
   [ 0 1 1 1 ]
   [ 1 0 1 0 ]
   [ 1 1 0 0 ]
   [ 1 0 0 0 ]

 Subgraph on 3 vertices:
   [ 0 1 1 ]
   [ 1 0 1 ]
   [ 1 1 0 ]

 Subgraph on 5 vertices:
   [ 0 1 0 0 1 ]
   [ 1 0 1 0 0 ]
   [ 0 1 0 1 0 ]
   [ 0 0 1 0 1 ]
   [ 1 0 0 1 0 ]

 Subgraph on 3 vertices:
   [ 0 1 1 ]
   [ 1 0 1 ]
   [ 1 1 0 ]

 Subgraph on 5 vertices:
   [ 0 1 1 1 1 ]
   [ 1 0 0 0 0 ]
   [ 1 0 0 0 0 ]
   [ 1 0 0 0 0 ]
   [ 1 0 0 0 0 ]

 Subgraph on 2 vertices:
   [ 0 1 ]
   [ 1 0 ]
