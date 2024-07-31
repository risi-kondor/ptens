############
Installation
############

`ptens` is installed as a PyTorch C++ (or CUDA) extension and requires the following: 

#. C++17 or higher
#. PyTorch
#. `cnine` (see below) 

ptens is easiest to install with ``pip``:

#. Download the `cnine <https://github.com/risi-kondor/cnine>`_  library 
#. Download `ptens` from `github <https://github.com/risi-kondor/ptens>`_. 
   By default, it is assumed that cnine and ptens are downloaded to the same directory 
   (e.g., ``Downloads``).      
#. Edit the user configurable variables in ``python/setup.py`` as necessary. 
#. Run the command ``pip install -e .`` in the ``ptens/python`` directory. 

To use `ptens` from Python, load the module the usual way with ``import ptens``. 


*************
Configuration
*************

The `ptens` installation can be configured by setting the corresponding variables in ``python/setup.py``.

``compile_with_cuda``
  If set to ``True``, `ptens` will be compiled with GPU suport. This requires a working CUDA and CUBLAS installation 
  and PyTorch itself having been compiled with CUDA enabled. 

.. 
  To make sure that the appropriate 
  runtime libraries are loaded, you must always import ``torch`` before importing ``ptens``.

