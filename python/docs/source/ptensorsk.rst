**************
Ptensor layers
**************

In most applications, Ptensors are organized into layers, captured by the classes 
``Ptensors0``, ``Ptensors1`` and ``Ptensors2``.  
One of the keys to efficient message passing is that `ptens` can operate in parallel 
on all the Ptensors in a given layer, even if their reference domains are of different sizes. 
Defining Ptensor layers requires specifying the reference domain of each Ptensor ising the 
``atomslist`` class.

==========
Atoms list
==========





