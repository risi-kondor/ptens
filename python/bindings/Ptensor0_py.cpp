

pybind11::class_<Ptensor0>(m,"ptensor0")

//.def(pybind11::init<const Ptensor0&, const Atoms&, const int, const FILLTYPE& dummy, const int _dev>(),"")

  .def_static("raw",[](const Atoms& _atoms, const int _nc, const int _dev){
      return Ptensor0::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("view",[](at::Tensor& x, const vector<int>& v){return Ptensor0::view(x,v);})

  .def("add_linmaps0",[](Ptensor0& obj, const Ptensor0& x, int offs){
      return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)

  .def("str",&Ptensor0::str,py::arg("indent")="")
  .def("__str__",&Ptensor0::str,py::arg("indent")="");


