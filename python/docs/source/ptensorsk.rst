**************
Ptensor layers
**************

In most applications, Ptensors are organized into layers, represented by the 
``ptensors0``, ``ptensors1`` and ``tensors2`` classes.  
A key feature of `ptens` is that it can operate  
on all the Ptensors in a given layer *in parallel*, even if their reference domains are of different sizes. 

=======================
Defining Ptensor layers
=======================

Similarly to individual Ptensors, the Ptensor layers classes also provide  
``zero``, ``randn`` or ``sequential`` constructors.  
For example, the following creates a layer consisting of three 
random first order Ptensors with reference domains :math:`(1,2,3)`, :math:`(3,5)` and :math:`(2)`
and 3 channels: 

.. code-block:: python
 
 >>> A=ptens.ptensors1.randn([[1,2,3],[3,5],[2]],3)
 >>> A
 Ptensor1(1,2,3):
 [ -1.23974 -0.407472 1.61201 ]
 [ 0.399771 1.3828 0.0523187 ]
 [ -0.904146 1.87065 -1.66043 ]
 
 Ptensor1(3,5):
 [ 0.0757219 1.47339 0.097221 ]
 [ -0.89237 -0.228782 1.16493 ] 
 
 Ptensor1(2):
 [ 0.584898 -0.660558 0.534755 ]


Unlike individual Ptensors, the ``ptensors0``, ``ptensors1`` and ``tensors2`` classes 
are not derived classes of ``torch.Tensor``. For 0'th order Ptensor layers, however, it 
is possible to define the layer   
from an :math:`N\times c` dimensional PyTorch tensor, where :math:`c` is the number of channels:

.. code-block:: python

 >>> M=torch.randn(3,5)
 >>> M
 tensor([[-1.2385, -0.4237, -1.2900, -1.4475,  0.2929],
         [-0.0483,  0.8409,  0.3700,  0.5826, -1.2325],
         [ 1.8040, -0.1950, -1.4181, -0.7805, -0.6050]])
 >>> A=ptens.ptensors0.from_matrix(M)
 >>> A
 Ptensor0 [0]:
 [ -1.23848 -0.423698 -1.28997 -1.44752 0.292851 ] 

 Ptensor0 [1]:
 [ -0.0482669 0.840887 0.370005 0.58262 -1.23249 ] 

 Ptensor0 [2]:
 [ 1.80402 -0.195021 -1.41805 -0.780468 -0.604952 ]


Conversely, the ``torch()`` method of ``ptensors0`` returns the content of the layer in a single 
PyTorch tensor:

.. code-block:: python

 >>> A=ptens.ptensors0.sequential([[1],[2],[3]],5)
 >>> B=A.torch()
 >>> A
 Ptensor0 [1]:
 [ 0 0 0 0 0 ] 

 Ptensor0 [2]:
 [ 1 1 1 1 1 ] 

 Ptensor0 [3]:
 [ 2 2 2 2 2 ]
 
 >>> B
 tensor([[0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1.],
         [2., 2., 2., 2., 2.]])

Similarly to individual Ptensors, Ptensor layers can be created on the GPU by adding a ``device`` 
argument to their constructor and can be moved to/from the GPU using the ``to`` method. 
All operations on GPU-resident layers are performed on the GPU.


===================
Getters and setters
===================

For higher order tensor layers, the individual tensors have to be accessed one by one if they are to 
be converted to a Pytorch tensor, for example:

.. code-block:: python

 >>> A=ptens.ptensors1.randn([[1,2],[2,3],[3]],5)
 >>> A
 Ptensor1 [1,2]:
 [ -1.23974 -0.407472 1.61201 0.399771 1.3828 ]
 [ 0.0523187 -0.904146 1.87065 -1.66043 -0.688081 ]
 
 Ptensor1 [2,3]:
 [ 0.0757219 1.47339 0.097221 -0.89237 -0.228782 ]
 [ 1.16493 0.584898 -0.660558 0.534755 -0.607787 ] 

 Ptensor1 [3]:
 [ 0.74589 -1.75177 -0.965146 -0.474282 -0.546571 ]
 
 >>> B=A[1]
 >>> B
 Ptensor1 [2,3]:
 [ 0.0757219 1.47339 0.097221 -0.89237 -0.228782 ]
 [ 1.16493 0.584898 -0.660558 0.534755 -0.607787 ]
 

Accessing individual tensors, as well as the constructor and ``torch()`` methods for ``ptensors0`` 
described above are differentiable operations.


========================================
Equivariant operations on Ptensor layers
========================================

Because the Ptensor layers are not subclasses of  ``torch.Tensor``, they do not automatically inherit all the 
usual arithmetic operations like addition multiplication by scalars, etc.. 
Currently, four basic operations are implemented for these classes: addition, concatenation,  
multiplication by matrices, and the ReU operator. 
All three of these operations are equivariant and implemented 
in a way that supports backpropagating gradients through them. 

--------------------------
Addition and concatenation
--------------------------

Two matching Ptensor layers (i.e., two layers such that the i'th tensor in the first layer has the 
same order, same number of channels and same reference domain as the i'th tensor in the second layer) 
can be added together:

.. code-block:: python

 >>> A=ptens.ptensors1.randn([[1,2],[2,3],[3]],3)
 >>> B=ptens.ptensors1.randn([[1,2],[2,3],[3]],3)
 >>> C=A+B
 >>> C
 Ptensor1 [1,2]:
 [ 1.44739 0.556559 -1.06723 ]
 [ -0.586973 2.43145 1.42343 ]

 Ptensor1 [2,3]:
 [ -3.47165 0.924936 -1.3852 ]
 [ -0.556994 -1.03874 0.25647 ]

 Ptensor1 [3]:
 [ -1.96103 -0.993459 1.36575 ]


Two matching Ptensor layers can also be concatenated along their channel dimension:

.. code-block:: python

 >>> A=ptens.ptensors1.randn([[1,2],[2,3],[3]],3)
 >>> B=ptens.ptensors1.randn([[1,2],[2,3],[3]],3)
 >>> C=ptens.cat(A,B)
 >>> C
 Ptensor1 [1,2]:
 [ 0.584898 -0.660558 0.534755 -1.23974 -0.407472 1.61201 ]
 [ -0.607787 0.74589 -1.75177 0.399771 1.3828 0.0523187 ] 

 Ptensor1 [2,3]:
 [ -0.965146 -0.474282 -0.546571 -0.904146 1.87065 -1.66043 ]
 [ -0.0384917 0.194947 -0.485144 -0.688081 0.0757219 1.47339 ] 
 
 Ptensor1 [3]:
 [ -0.370271 -1.12408 1.73664 0.097221 -0.89237 -0.228782 ]



--------------------------
Multiplication by matrices
--------------------------

Multiplying Ptensors by matrices along their channel dimension is an equivariant operation. 
The primary way that learnable 
parameters are introduced in permutation equivariant nets is in the form of such mixing matrices.
The following example demostrates this for a ``ptensors1`` object, but the same works for 
``ptensors0`` and ``ptensors2`` layers as well. 

.. code-block:: python

 >>> A=ptens.ptensors1.randn([[1,2],[2,3],[3]],5)
 >>> A
 Ptensor1 [1,2]:
 [ -0.274068 0.005616 -1.77286 0.519691 0.0431933 ]
 [ -1.96668 -0.480737 -1.83641 -0.257851 -0.391737 ]

 Ptensor1 [2,3]:
 [ 2.69588 1.6585 -1.13769 -1.22027 0.111152 ]
 [ -0.672931 -1.39814 -0.477463 0.643125 1.37519 ]

 Ptensor1 [3]:
 [ -1.2589 0.259477 -1.6247 -0.996947 -0.149277 ]

 >>> M=torch.randn(5,2)
 >>> B=A*M
 >>> B
 Ptensor1 [1,2]:
 [ -0.164324 -2.41585 ]
 [ -3.48671 -0.161725 ]

 Ptensor1 [2,3]:
 [ 2.90595 -7.82352 ]
 [ 0.69117 3.77597 ]

 Ptensor1 [3]:
 [ -2.84689 -1.14514 ]


----
ReLU
----

The ``relu(x,alpha)`` applies the function :math:`sigma(x)=\textrm{max}(x,\alpha x)` 
(with :math:`0\leq \alpha<1`) elementwise and  can be applied to Ptensor layers of any order.

.. code-block:: python

 >>> A=p.ptensors0.randn(3,3)
 >>> print(A)
 Ptensor0 [0]:
 [ -1.23974 -0.407472 1.61201 ]

 Ptensor0 [1]:
 [ 0.399771 1.3828 0.0523187 ]

 Ptensor0 [2]:
 [ -0.904146 1.87065 -1.66043 ]

 >>> B=p.relu(A,0.1)
 >>> print(B)
 Ptensor0 [0]:
 [ -0.123974 -0.0407472 1.61201 ]

 Ptensor0 [1]:
 [ 0.399771 1.3828 0.0523187 ]

 Ptensor0 [2]:
 [ -0.0904147 1.87065 -0.166043 ]
















