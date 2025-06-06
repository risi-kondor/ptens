pybind11::class_<BatchedAtomsPack<2> >(m,"batched_atomspack2")

  .def(py::init([](const vector<AtomsPack>& x){return BatchedAtomsPack<2>(x);}))
  .def(py::init([](const vector<vector<vector<int> > >& x){return BatchedAtomsPack<2>(x);}))

  .def("__len__",&BatchedAtomsPack<2>::size)
  .def("__getitem__",[](const BatchedAtomsPack<2>& x, const int i){return x[i];})
  .def("torch",[](const BatchedAtomsPack<2>& x){return x.as_vecs();})

  .def("nrows0",[](const BatchedAtomsPack<2>& x){return x.nrows0();})
  .def("nrows1",[](const BatchedAtomsPack<2>& x){return x.nrows1();})
  .def("nrows2",[](const BatchedAtomsPack<2>& x){return x.nrows2();})

  .def("nrows0",[](const BatchedAtomsPack<2>& x, const int i){return x.nrows0(i);})
  .def("nrows1",[](const BatchedAtomsPack<2>& x, const int i){return x.nrows1(i);})
  .def("nrows2",[](const BatchedAtomsPack<2>& x, const int i){return x.nrows2(i);})

  .def("offset0",[](const BatchedAtomsPack<2>& x, const int i){return x.offset0(i);})
  .def("offset1",[](const BatchedAtomsPack<2>& x, const int i){return x.offset1(i);})
  .def("offset2",[](const BatchedAtomsPack<2>& x, const int i){return x.offset2(i);})

  .def("str",&BatchedAtomsPack<2>::str,py::arg("indent")="")
  .def("__str__",&BatchedAtomsPack<2>::str,py::arg("indent")="");





