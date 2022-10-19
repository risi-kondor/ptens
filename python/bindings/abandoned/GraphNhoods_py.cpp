
pybind11::class_<Nhoods>(m,"nhoods")

  .def("str",&Nhoods::str,py::arg("indent")="")
  .def("__str__",&Nhoods::str,py::arg("indent")="");



pybind11::class_<GraphNhoods>(m,"graphNhoods")

  .def(pybind11::init<const Cgraph&, const int>())

  .def("level",&GraphNhoods::level);

