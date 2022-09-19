***********************
Equivariant linear maps
***********************

The neurons in permutation equivariant networks correspond to learnable equivariant mappings 
between Ptensors. `ptens` implements all such possible linear maps. 
We start with the case where the reference domain of the input and output are the same. 

==========
Order zero
==========

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

==========
Order one
==========

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

==========
Order two
==========

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

