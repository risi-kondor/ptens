**************
Ptensor layers
**************

In most applications, Ptensors are organized into layers, represented by the 
``ptensorlayer0``, ``ptensorlayer1`` and ``ptensorlayer2`` classes.  
A key feature of `ptens` is that when the Ptensor layers are on the GPU, 
all operations on them are parallelized across the individual Ptensors that they contain, 
even if the reference domains of the individual Ptensors are of different sizes. 

=======================
Defining Ptensor layers
=======================

The reference domains of a Ptensor layer are stored in a ``ptens_base.atomspack`` object, for example: 

.. code-block:: python

  >> atoms=ptens_base.atomspack([[1,2,3],[3,5],[2]])
  >> print(atoms)

  ([1,2,3],[3,5],[2])

Similary to individual Ptensors, a ``patensorlayer`` can be created using the 
``zero``, ``randn`` or ``sequential`` constructors: 

.. code-block:: python
 
 >> A=ptens.ptensors1.randn(atoms,3)
 >> print(A)

 Ptensorlayer1:
   Ptensor1 [1,2,3]:
     [ -2.0702 1.16687 -0.260996 ]
     [ 1.19406 -2.29932 0.0684815 ]
     [ -1.36043 0.906236 0.386861 ]
   Ptensor1 [3,5]:
     [ -0.289209 1.05266 -1.15485 ]
     [ 0.519951 -0.263112 -0.317281 ]
   Ptensor1 [2]:
     [ -1.11236 1.03108 0.739697 ]

For convenience, the layer can also be constructed directly from the list of reference domains:

.. code-block:: python
 
 >> ptens.ptensors1.randn([[1,2,3],[3,5],[2]],3)
 >> print(A)

 Ptensorlayer1:
   Ptensor1 [1,2,3]:
     [ 0.772723 1.02357 1.41353 ]
     [ -0.585875 -0.162227 -0.0738079 ]
     [ -0.119266 -0.461233 0.19949 ]
   Ptensor1 [3,5]:
     [ -0.814475 0.372868 -0.970456 ]
     [ -1.02822 0.380239 -1.73501 ]
   Ptensor1 [2]:
     [ 1.26059 0.753664 -0.743881 ]

For ease of compatibility with PyTorch's own functionality and some other libraries, 
the ``ptensorlayer0``, ``ptensorlayer1`` and ``tensorlayer2`` classes are implemented as subclasses of 
``torch.Tensor`` and all the Ptensors in a given layer are stacked into a single matrix 
where the columns correspond to the channels:

.. code-block:: python

 >> print(torch.Tensor(A))

 tensor([[ 0.7727,  1.0236,  1.4135],
	 [-0.5859, -0.1622, -0.0738],
	 [-0.1193, -0.4612,  0.1995],
	 [-0.8145,  0.3729, -0.9705],
	 [-1.0282,  0.3802, -1.7350],
	 [ 1.2606,  0.7537, -0.7439]])

A Ptensor layer can also be constructed directly from this matrix: 

.. code-block:: python

 >> M=torch.randn(6,3)
 >> A=ptens.ptensorlayer1.from_matrix([[1,2,3],[3,5],[2]],M)
 >> print(A)

 Ptensorlayer1:
   Ptensor1 [1,2,3]:
     [ 0.0600628 0.966446 0.784876 ]
     [ 0.250401 1.13511 0.644161 ]
     [ -1.38752 0.81458 -0.711916 ]
   Ptensor1 [3,5]:
     [ -1.25401 -0.245323 -0.377335 ]
     [ 0.962375 1.16961 0.93007 ]
   Ptensor1 [2]:
     [ 0.385544 0.249942 0.250718 ]


Similarly to individual Ptensors, Ptensor layers can be created on the GPU by adding a ``device`` 
argument to their constructor and can be moved to/from the GPU using the ``to`` method. 
All operations on GPU-resident layers are performed on the GPU.

===================
Getters and setters
===================

Individual Ptensors in a given layer can be accessed by subscripting:

.. code-block:: python

 >> print(A[1])

 Ptensor1 [3,5]:
   [ -1.25401 -0.245323 -0.377335 ]
   [ 0.962375 1.16961 0.93007 ]


========================================
Equivariant operations on Ptensor layers
========================================

The fact that Ptensor layers are stored by stacking their individual Ptensors in a single matrix makes 
some common equivariant operations on them easy to implement. For example, linear layers 
simply correspond to matrix multiplication from the right, followed by adding constants to the columns, 
just like in many other standard architectures, allowing us to reuse PyTorch's ``linear`` module. 
Elementwise operations such as ``relu`` are equally easy to apply:

.. code-block:: python

 >> A=ptens.ptensorlayer1.randn([[1,2,3],[3,5],[2]],3)
 >> B=torch.relu(A)
 >> print(B)

 Ptensorlayer1:
   Ptensor1 [1,2,3]:
     [ 0 0.637496 0 ]
     [ 0 0 1.62583 ]
     [ 0.303279 0 0.15176 ]
   Ptensor1 [3,5]:
     [ 0 0 0.246751 ]
     [ 0 0.299123 1.52228 ]
   Ptensor1 [2]:
     [ 0 0.0121746 0.452276 ]

In general, any operation that returns a data structure that transforms as a Ptensor layer 
will return a ``ptensorlayer0``, ``ptensorlayer1`` or ``ptensorlayer2`` object, as appropriate. 
Operations that are equivariant but do not result in a Ptensors return an ordinary PyTorch tensor, for 
example:

.. code-block:: python

 >> A=ptens.ptensorlayer1.randn([[1,2,3],[3,5],[2]],3)
 >> print(torch.norm(A))

 tensor(2.5625)
 
==========
Atomspacks
==========

To implement operations on Ptensor layers that manipulate individual Ptensors or individual rows 
corresponding to specific elements of their reference domain it might be necessary to access the   
``atomspack`` object stored in the layer's ``atoms`` variable. 
The reference domain of the ``i``'th Ptensor can be extracted from the ``atomspack`` by subscripting:

.. code-block:: python

 >> print(A.atoms[1])

 [3, 5]

The number of rows allocated to the ``i``'th Ptensor and the corresponding row offset is accessed 
via the ``norws0``, ``nrows1``, ``nrows2`` and ``row_offset0``, ``row_offset1`` ` ``row_offset2`` methods 
respectively depending on whether the underlying object is a zeroth, first, or second order layer:

.. code-block:: python

 >> print(A.atoms.nrows1(1))
 >> print(A.atoms.row_offset1(1))

 2
 3





.. 
 ========================================
 Equivariant operations on Ptensor layers
 ========================================

 Because the Ptensor layers are not subclasses of  ``torch.Tensor``, they do not automatically inherit all the 
 usual arithmetic operations like addition multiplication by scalars, etc.. 
 Currently, four basic operations are implemented for these classes: addition, concatenation,  
 multiplication by matrices, and the ReU operator. 
 All three of these operations are equivariant and implemented 
 in a way that supports backpropagating gradients through them. 


