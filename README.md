# ptens

_ptens_ is a C++/CUDA library with a PyTorch front end interface for permutation equivariant message passing and 
constructing higher order equivariant graph neural networks based on the Ptensors formalism (https://arxiv.org/abs/2306.10767). 
On the GPU, the library can parallelize the message passing process across 
many pairs of Ptensors, even when the Ptensors are of different sizes and 
their reference domains overlap in different ways. 

_ptens_ is developed by Risi Kondor's research group and is released for noncommercial use 
under the license provided in LICENSE.txt. 

Documentation for the library's Python/PyTorch API can be found at https://risi-kondor.github.io/ptens-doc/.

## Installation

Ptens uses CPM to automatically download the [cnine](https://github.com/risi-kondor/cnine) dependency automatically.
If that is not your intent, check out [CPM's docs](https://github.com/cpm-cmake/CPM.cmake?tab=readme-ov-file#local-package-override) for your options.

## GPU / CUDA installation

Please install the CUDA toolkit first!

Install with pip (includes PyTorch 2.0+):
```bash
pip install .
```

## CPU installation

For custom PyTorch versions:
1. Install desired PyTorch CPU first
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
2. Then install the build dependencies of ptens manually. (Listed in `pyproject.toml`/`build-system`/`requires`.) Except for `torch` since we manually installed that first.
   ```bash
   pip install scikit-build-core pybind11
   ```
2. Then install ptens, we disable the isolation build, so that module we just manually installed are used as dependencies during the build stage
   ```bash
   pip install --no-build-isolation .
   ```

#### Manual CMake Build

Ptens uses scikit-build-core. To build manually:
```bash
git clone https://github.com/risi-kondor/ptens.git
cd ptens
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/python/installation"
make -j4
```
