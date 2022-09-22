**************
Ptensor layers
**************

In most applications, Ptensors are organized into layers, represented by the 
``ptensors0``, ``ptensors1`` and ``tensors2`` classes.  
One of the keys to efficient message passing is that `ptens` can operate  
on all the Ptensors in a given layer *in parallel*, even if their reference domains are of different sizes. 

=======================
Defining Ptensor layers
=======================

Similarly to individual Ptensors, one way to define Ptensor layers is using the ``zero``, 
``randn`` or ``sequential`` constructors. For example, the following creates a layer consisting of three 
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


===================
Getters and setters
===================


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

In higher order Ptensor layers, the tensors cannot be jointly converted to/from a single PyTorch tensor, 
since their dimensionalities might be different. 















