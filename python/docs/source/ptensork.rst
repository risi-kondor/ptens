********
Ptensors
********

A :math:`p`'th order *permutationally covariant tensor* or *Ptensor* for short, with  
reference domain :math:`(a_1,\ldots,a_k)` is a :math:`(p+1)`'th order tensor 
:math:`A\in\mathbb{R}^{k\times k\times\ldots\times k\times c}`, where :math:`c` is the number 
of channels. The elements of reference domain are called `atoms`. 
The defining property of Ptensors is that if :math:`(a_1,\ldots,a_k)` are permuted 
by a permutation :math:`\sigma`, then :math:`A` transforms to :math:`A'` with 

.. math::
  A'_{i_1,\ldots,i_k,c}=A_{i_{\sigma^{-1}(1)},\ldots,i_{\sigma^{-1}(k)},c}

Currently `ptens` supports zeroth, first and second order Ptensors with corresponding classes 
``ptensor0``, ``ptensor1`` and ``ptensor2``. Each of these classes is derived from the 
``torch.Tensor`` class, hence all the usual arithmetic operations can be applied to PTensors. 
Note that some of these operations might break equiavriance, though. For example, applying 
a specific operation to just one slice of a Ptensor (unless it is a slice corresponding to a 
given setting of the channel dimension) generally breaks equivariance. 

=================
Creating Ptensors
=================
 
Ptensors can be created with the familiar `zeros` or `randn` constructors. 
For example,

.. code-block:: python

 >>> A=ptens.ptensor0.randn([2],5)

creates a zeroth order PTensor with reference domain :math:`(2)` and 5 channels. 
Printing out the Ptensor returns both its contents and its reference domain:

.. code-block:: python

 >>> print(A)
 Ptensor0(2):
 [ -1.97856 -1.72226 -0.0215097 -2.61169 1.3889 ]

For higher order Ptensors, the size of the first :math:`p` dimensions is inferred from the 
size of the reference domain. For example, the following creates a first order Ptensor over 3 atoms:

.. code-block:: python

 >>> B=ptens.ptensor1.randn([1,2,3],5)
 >>> print(B)
 Ptensor1(1,2,3):
 [ 1.57658 0.278142 1.78178 ]
 [ 0.524821 -0.496081 -0.427595 ]
 [ -1.94122 -0.0458005 1.8405 ]
 [ 0.848817 0.209676 0.323065 ]
 [ -0.529721 -0.903987 0.081249 ]

Similalry, the following creates and prints out a second order Ptensor over the reference domain 
:math:`(1,2,3)`:

.. code-block:: python

 >>> C=ptens.ptensor2.randn([1,2,3],5)
 >>> print(C)
 Ptensor2(1,2,3):
 channel 0:
   [ 0.619967 0.703344 0.161594 ]
   [ -1.07889 1.21051 0.247078 ]
   [ 0.0626437 -1.48677 -0.117047 ]

 channel 1:
   [ -0.809459 0.768829 0.80504 ]
   [ 0.69907 -0.824901 0.885139 ]
   [ 1.45072 -2.47353 -1.03353 ]

 channel 2:
   [ -0.481529 -0.240306 2.9001 ]
   [ 1.07718 -0.507446 1.1044 ]
   [ 1.5038 -1.10569 0.210451 ]

 channel 3:
   [ -0.172885 0.117831 -0.62321 ]
   [ 0.201925 -0.486807 0.0418346 ]
   [ 0.041158 1.72335 -0.199498 ]

 channel 4:
   [ 0.375979 3.05989 1.30477 ]
   [ -1.76276 -0.139075 -0.349366 ]
   [ -0.0366747 -0.563576 0.233288 ]

Note that for better legibility the `ptensor` classes use their own custom printout methods. 

For debugging purposes `ptens` also provides a ``sequential`` initializer for each of these classes, e.g.:

.. code-block:: python

 >>> A=ptens.ptensor1.sequential([1,2,3],5)
 >>> print(A)
 Ptensor1(1,2,3):
 [ 0 5 10 ]
 [ 1 6 11 ]
 [ 2 7 12 ]
 [ 3 8 13 ]
 [ 4 9 14 ]


=======================
Equivariant linear maps
=======================

The neurons in permutation equivariant networks correspond to learnable equivariant mappings 
between Ptensors. `ptens` implements all such possible linear maps. 
We start with the case where the reference domain of the input and output are the same. 

--------
ptensor0
--------

The only equivariant linear maps between zeroth order Ptensors are multiples of the identity, 
therefore ``linmaps0`` applied to a ``ptensor0``, is just the identity mapping:

.. code-block:: python

 >>> A=ptens.ptensor0.sequential([1,2,3],5)
 >>> print(A)
 Ptensor0(1,2,3):
 [ 0 1 2 3 4 ]
 >>> B=ptens.linmaps0(A)
 >>> print(B)
 Ptensor0(1,2,3):
 [ 0 1 2 3 4 ]

The only equivariant map from a ``ptensor0`` to ``ptensor1`` is given by :math:`B_{i,c}=A_c`:

.. code-block:: python

 >>> A=ptens.ptensor0.sequential([1,2,3],3)
 >>> print(A)
 Ptensor0(1,2,3):
 [ 0 1 2 ]
 >>> B=ptens.linmaps1(A)
 >>> print(B)
 Ptensor1(1,2,3):
 [ 0 1 2 ]
 [ 0 1 2 ]
 [ 0 1 2 ]

On the other hand, the space of equivariant maps from a ``ptensor0`` to ``ptensor1`` are 
spanned by two differrent maps: :math:`C^1_{i,j,c}=A_c` and :math:`C^2_{i,j,c}=\delta_{i,j} A_c`. 
Consequently, `linmaps2` doubles the number of channels:

.. code-block:: python

 >>> A=ptens.ptensor0.sequential([1,2,3],3)
 >>> print(A)
 Ptensor0(1,2,3):
 [ 0 1 2 ]

 >>> C=ptens.linmaps2(A)
 >>> print(C)
 Ptensor2(1,2,3):
 channel 0:
   [ 0 0 0 ]
   [ 0 0 0 ]
   [ 0 0 0 ]

 channel 1:
   [ 1 1 1 ]
   [ 1 1 1 ]
   [ 1 1 1 ]

 channel 2:
   [ 2 2 2 ]
   [ 2 2 2 ]
   [ 2 2 2 ]

 channel 3:
   [ 0 0 0 ]
   [ 0 0 0 ]
   [ 0 0 0 ]

 channel 4:
   [ 1 0 0 ]
   [ 0 1 0 ]
   [ 0 0 1 ]

 channel 5:
   [ 2 0 0 ]
   [ 0 2 0 ]
   [ 0 0 2 ]


--------
ptensor1
--------

When mapping a first order Ptensor to a zeroth order Ptensor, the only equivariant linear map 
is :math:`B_c=\sum_i A_{i,c}`:

.. code-block:: python

 >>> A=ptens.ptensor1.sequential([1,2,3],3)
 >>> print(A)
 Ptensor1(1,2,3):
 [ 0 1 2 ]
 [ 3 4 5 ]
 [ 6 7 8 ]
 >>> B=ptens.linmaps0(A)
 >>> print(B)
 Ptensor0(1,2,3):
 [ 9 12 15 ]

On the other hand, there are two ways of mapping a first order Ptensor to a first order Ptensor: 
:math:`B_{i,c}=\sum_i A_{i,c}` and :math:`B_{i,c}=A_{i,c}`. Therefore, the number of channels doubles: 

.. code-block:: python

 >>> A=ptens.ptensor1.sequential([1,2,3],3)
 >>> print(A)
 Ptensor1(1,2,3):
 [ 0 1 2 ]
 [ 3 4 5 ]
 [ 6 7 8 ]
 >>> B=ptens.linmaps1(A)
 >>> print(B)
 Ptensor1(1,2,3):
 [ 9 12 15 0 1 2 ]
 [ 9 12 15 3 4 5 ]
 [ 9 12 15 6 7 8 ]

There are a total of five equivariant maps from a first order Ptensor to a second order Ptensor: 
:math:`B_{i',j',c}=\sum_i A_{i,c}`, 
:math:`B_{i',j',c}=\delta_{i',j'} \sum_i A_{i,c}`, 
:math:`B_{i,j,c}=A_{i,c}`, 
:math:`B_{j,i,c}=A_{i,c}` and 
:math:`B_{i,j,c}=\delta_{i,j} A_{i,c}`. 
Hence the number of channels multiplies fivefold. 

.. code-block:: python

 >>> A=ptens.ptensor1.sequential([1,2,3],3)
 >>> print(A)
 Ptensor1(1,2,3):
 [ 0 1 2 ]
 [ 3 4 5 ]
 [ 6 7 8 ]

 >>> B=ptens.linmaps2(A)
 >>> print(B)
 Ptensor2(1,2,3):
 channel 0:
   [ 9 9 9 ]
   [ 9 9 9 ]
   [ 9 9 9 ]

 channel 1:
   [ 10 10 10 ]
   [ 10 10 10 ]
   [ 10 10 10 ]

 channel 2:
   [ 15 15 15 ]
   [ 15 15 15 ]
   [ 15 15 15 ]

 channel 3:
   [ 9 0 0 ]
   [ 0 9 0 ]
   [ 0 0 9 ]

 channel 4:
   [ 10 0 0 ]
   [ 0 10 0 ]
   [ 0 0 10 ]

 channel 5:
   [ 15 0 0 ]
   [ 0 15 0 ]
   [ 0 0 15 ]

 channel 6:
   [ 0 3 6 ]
   [ 0 3 6 ]
   [ 0 3 6 ]

 channel 7:
   [ 1 4 7 ]
   [ 1 4 7 ]
   [ 1 4 7 ]

 channel 8:
   [ 2 5 8 ]
   [ 2 5 8 ]
   [ 2 5 8 ]

 channel 9:
   [ 0 0 0 ]
   [ 3 3 3 ]
   [ 6 6 6 ]

 channel 10:
   [ 1 1 1 ]
   [ 4 4 4 ]
   [ 7 7 7 ]

 channel 11:
   [ 2 2 2 ]
   [ 5 5 5 ]
   [ 8 8 8 ]

 channel 12:
   [ 0 0 0 ]
   [ 0 3 0 ]
   [ 0 0 6 ]

 channel 13:
   [ 1 0 0 ]
   [ 0 4 0 ]
   [ 0 0 7 ]

 channel 14:
   [ 2 0 0 ]
   [ 0 5 0 ]
   [ 0 0 8 ]


--------
ptensor2
--------

The space of equivariant maps from a second order Ptensor to a zeroth order Ptensor is spanned by 
:math:`B^1_{c}=\sum_i \sum_j A_{i,j,c}` and 
:math:`B^2_{c}=\sum_i A_{i,i,c}`. 


.. code-block:: python

 >>> A=ptens.ptensor2.sequential([1,2,3],3)
 >>> print(A)
 Ptensor2(1,2,3):
 channel 0:
   [ 0 3 6 ]
   [ 9 12 15 ]
   [ 18 21 24 ]

 channel 1:
   [ 1 4 7 ]
   [ 10 13 16 ]
   [ 19 22 25 ]

 channel 2:
   [ 2 5 8 ]
   [ 11 14 17 ]
   [ 20 23 26 ]

 >>> B=ptens.linmaps0(A)
 >>> print(B)
 Ptensor0(1,2,3):
 [ 108 117 126 36 39 42 ]

The space of equivariant maps from a second order Ptensor to a first order Ptensor is spanned by 
:math:`B^1_{i',c}=\sum_i \sum_j A_{i,j,c}`, 
:math:`B^2_{i',c}=\sum_i A_{i,i,c}`,
:math:`B^3_{i,c}=\sum_j A_{i,j,c}`, 
:math:`B^4_{i,c}=\sum_j A_{j,i,c}`, and  
:math:`B^5_{i,c}=\sum_j A_{i,i,c}`. 

.. code-block:: python

 >>> B=ptens.linmaps1(A)
 >>> print(B)
 Ptensor1(1,2,3):
 [ 108 117 126 36 39 42 27 30 33 9 12 15 0 1 2 ]
 [ 108 117 126 36 39 42 36 39 42 36 39 42 12 13 14 ]
 [ 108 117 126 36 39 42 45 48 51 63 66 69 24 25 26 ]


The space of equivariant maps from a second order Ptensor to a second order Ptensor is spanned by 
15 different maps (output truncated). 

.. code-block:: python

 >>> B=ptens.linmaps2(A)
 >>> print(B)
 Ptensor2(1,2,3):
 channel 0:
   [ 108 108 108 ]
   [ 108 108 108 ]
   [ 108 108 108 ]

 channel 1:
   [ 117 117 117 ]
   [ 117 117 117 ]
   [ 117 117 117 ]

 channel 2:
   [ 126 126 126 ]
   [ 126 126 126 ]
   [ 126 126 126 ]

 channel 3:
   [ 36 36 36 ]
   [ 36 36 36 ]
   [ 36 36 36 ]

 channel 4:
   [ 39 39 39 ]
   [ 39 39 39 ]
   [ 39 39 39 ]

 channel 5:
   [ 42 42 42 ]
   [ 42 42 42 ]
   [ 42 42 42 ]

 channel 6:
   [ 108 0 0 ]
   [ 0 108 0 ]
   [ 0 0 108 ]

 channel 7:
   [ 117 0 0 ]
   [ 0 117 0 ]
   [ 0 0 117 ]








 