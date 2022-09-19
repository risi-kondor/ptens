pybind11::class_<Hgraph>(m,"graph")

  .def_static("random",static_cast<Hgraph(*)(const int, const float)>(&Hgraph::random))

  .def("nhoods",&Cgraph::nhoods)
  .def("set",&Cgraph::set)

  .def("str",&Cgraph::str,py::arg("indent")="")
  .def("__str__",&Cgraph::str,py::arg("indent")="");


