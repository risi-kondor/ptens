pybind11::class_<BatchedAtomsPack>(m,"batched_atomspack")

  .def(py::init([](const vector<AtomsPack>& x){return BatchedAtomsPack(x);}))
  .def(py::init([](const vector<vector<vector<int> > >& x){return BatchedAtomsPack(x);}))

  .def("__len__",&BatchedAtomsPack::size)
  .def("__getitem__",[](const BatchedAtomsPack& x, const int i){return x[i];})
  .def("torch",[](const BatchedAtomsPack& x){return x.as_vecs();})

  .def("str",&BatchedAtomsPack::str,py::arg("indent")="")
  .def("__str__",&BatchedAtomsPack::str,py::arg("indent")="");





