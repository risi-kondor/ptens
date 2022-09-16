

pybind11::class_<Ptensors2>(m,"ptensors2")

//.def(pybind11::init<const Ptensor0&, const Atoms&, const int, const FILLTYPE& dummy, const int _dev>(),"")

  .def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors2::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors2::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
//      return Ptensors2::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def("get_nc",&Ptensors2::get_nc)
  .def("atoms_of",&Ptensors2::atoms_of)
  .def("push_back",&Ptensors2::push_back)

  .def("str",&Ptensors2::str,py::arg("indent")="")
  .def("__str__",&Ptensors2::str,py::arg("indent")="");


