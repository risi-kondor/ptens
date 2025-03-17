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

### Basic Installation  
Create and activate a virtual environment:  
```bash
python -m venv /path/to/venv
source /path/to/venv/bin/activate
```

Install with pip (includes PyTorch 2.0+):  
```bash
pip install .
```

#### Advanced Installation

For custom PyTorch versions:  
1. Install desired PyTorch first:  
   ```bash
   pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```
2. Then install cnine:  
   ```bash
   pip install .
   ```

If you want to change options for the build, like activating or deactivating a CUDA build.
Either modify locally the `pyproject.toml` `cmake.define` section or export the appropriate environment variables.

This applies using a local version of `cnine` as well.
If you want to use a local `cnine` version instead of the current github version add `CPM_cnine_SOURCE="/path/to/cnine/"` to the `cmake.define` section.
CPM is the toolkit that we use to fetch `cnine` from github.

#### Manual CMake Build

Cnine uses scikit-build-core. To build manually:  
```bash
git clone https://github.com/risi-kondor/ptens
cd cnine
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/torch"
make -j4
```

