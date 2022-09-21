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

Currently `ptens` supports zeroth, first and second order Ptensors. thecorresponding classes 
``ptensor0``, ``ptensor1`` and ``ptensor2``. Each of these classes is derived  
``torch.Tensor``, allowing all the usual PyTorch arithmetic operations to be applied to PTensors. 
Note, however, that some of these operations might break equivariance. For example, changing 
just one slice or one element of a Ptensor is generally not an equivariant 
operation. 

.. 
  (unless it is a slice corresponding to a 
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
 [ 0.0515154 -0.0194946 -1.39105 -1.38258 0.658819 ]
 [ 0.85989 0.278101 0.890897 -0.000561227 1.54719 ]
 [ 1.22424 -0.099083 -0.849395 -0.396878 -0.119167 ]

Similarly, the following creates and prints out a second order Ptensor over the reference domain 
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

..
  Note that for better legibility the `ptensor` classes use their own custom printout methods. 

For debugging purposes `ptens` also provides a ``sequential`` initializer for each of these classes, e.g.:

.. code-block:: python

 >>> A=ptens.ptensor1.sequential([1,2,3],5)
 >>> print(A)
 Ptensor1(1,2,3):
 [ 0 1 2 3 4 ]
 [ 5 6 7 8 9 ]
 [ 10 11 12 13 14 ]









 