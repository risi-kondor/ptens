pybind11::class_<AtomsPack>(m,"atomspack")

  .def(py::init([](const vector<vector<int> >& x){return AtomsPack(x);}))
  .def_static("from_list",[](const vector<vector<int> >& x){return AtomsPack(x);})
  .def_static("random",[](const int n, const int m, const float p){return AtomsPack::random(n,m,p);})

  .def("__len__",&AtomsPack::size)
  .def("__eq__",&AtomsPack::operator==)
  .def("__getitem__",[](const AtomsPack& x, const int i){return vector<int>(x[i]);})
  .def("torch",[](const AtomsPack& x){return x.as_vecs();})

  .def("is_constk",[](const AtomsPack& x){return x.constk()>0;})
  .def("constk",[](const AtomsPack& x){return x.constk();})

  .def("nrows0",[](const AtomsPack& x){return x.nrows0();})
  .def("nrows1",[](const AtomsPack& x){return x.nrows1();})
  .def("nrows2",[](const AtomsPack& x){return x.nrows2();})

  .def("nrows0",[](const AtomsPack& x, const int i){return x.nrows0(i);})
  .def("nrows1",[](const AtomsPack& x, const int i){return x.nrows1(i);})
  .def("nrows2",[](const AtomsPack& x, const int i){return x.nrows2(i);})

  .def("row_offset0",[](const AtomsPack& x, const int i){return x.row_offset0(i);})
  .def("row_offset1",[](const AtomsPack& x, const int i){return x.row_offset1(i);})
  .def("row_offset2",[](const AtomsPack& x, const int i){return x.row_offset2(i);})

//.def("overlaps",[](const AtomsPack& x, const AtomsPack& y, const int min_overlaps=1){
//      return x.overlaps_mlist(y,min_overlaps);})


  .def("str",&AtomsPack::str,py::arg("indent")="")
  .def("__str__",&AtomsPack::str,py::arg("indent")="")
  .def("__repr__",&AtomsPack::str,py::arg("indent")="");





