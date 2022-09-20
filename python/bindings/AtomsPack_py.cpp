pybind11::class_<AtomsPack>(m,"atomslist")

  .def(py::init([](const vector<vector<int> >& x){return AtomsPack(x);}))

  .def_static("from_list",[](const vector<vector<int> >& x){return AtomsPack(x);})

  .def("str",&AtomsPack::str,py::arg("indent")="")
  .def("__str__",&AtomsPack::str,py::arg("indent")="")
  .def("__repr__",&AtomsPack::str,py::arg("indent")="");





