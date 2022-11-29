pybind11::class_<AtomsPack>(m,"atomspack")

  .def(py::init([](const vector<vector<int> >& x){return AtomsPack(x);}))

  .def_static("from_list",[](const vector<vector<int> >& x){return AtomsPack(x);})
  .def_static("random",[](const int n, const float p){return AtomsPack::random(n,p);})

  .def("__len__",&AtomsPack::size)
  .def("__getitem__",[](const AtomsPack& x, const int i){return vector<int>(x[i]);})

  .def("str",&AtomsPack::str,py::arg("indent")="")
  .def("__str__",&AtomsPack::str,py::arg("indent")="")
  .def("__repr__",&AtomsPack::str,py::arg("indent")="");





