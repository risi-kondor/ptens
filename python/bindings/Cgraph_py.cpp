

pybind11::class_<Cgraph>(m,"cgraph")

  .def_static("random",static_cast<Cgraph(*)(const int, const float)>(&Cgraph::random))

  .def("push",&Cgraph::push)

  .def("str",&Cgraph::str,py::arg("indent")="")
  .def("__str__",&Cgraph::str,py::arg("indent")="");

