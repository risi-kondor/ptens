pybind11::class_<BatchedAtomsPack<0> >(m,"batched_atomspack0")

  .def(py::init([](const vector<AtomsPack>& x){return BatchedAtomsPack<0>(x);}))
  .def(py::init([](const vector<vector<vector<int> > >& x){return BatchedAtomsPack<0>(x);}))

  .def("__len__",&BatchedAtomsPack<0>::size)
  .def("__getitem__",[](const BatchedAtomsPack<0>& x, const int i){return x[i];})
  .def("torch",[](const BatchedAtomsPack<0>& x){return x.as_vecs();})

  .def("nrows0",[](const BatchedAtomsPack<0>& x){return x.nrows0();})
  .def("nrows1",[](const BatchedAtomsPack<0>& x){return x.nrows1();})
  .def("nrows2",[](const BatchedAtomsPack<0>& x){return x.nrows2();})

  .def("nrows0",[](const BatchedAtomsPack<0>& x, const int i){return x.nrows0(i);})
  .def("nrows1",[](const BatchedAtomsPack<0>& x, const int i){return x.nrows1(i);})
  .def("nrows2",[](const BatchedAtomsPack<0>& x, const int i){return x.nrows2(i);})

  .def("offset0",[](const BatchedAtomsPack<0>& x, const int i){return x.offset0(i);})
  .def("offset1",[](const BatchedAtomsPack<0>& x, const int i){return x.offset1(i);})
  .def("offset2",[](const BatchedAtomsPack<0>& x, const int i){return x.offset2(i);})

  .def("str",&BatchedAtomsPack<0>::str,py::arg("indent")="")
  .def("__str__",&BatchedAtomsPack<0>::str,py::arg("indent")="");





