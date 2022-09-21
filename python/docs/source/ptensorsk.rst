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

Unlike individual Ptensors, however, the ``ptensors0``, ``ptensors1`` and ``tensors2`` classes 
are not derived classes of ``torch.Tensor``. Therefore the individual Ptensors in a ptensor layer 
cannot be directly accessed. 












