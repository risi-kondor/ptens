pybind11::class_<AtomsPack>(m,"atomslist")

  .def("str",&AtomsPack::str,py::arg("indent")="")
  .def("__str__",&AtomsPack::str,py::arg("indent")="");





