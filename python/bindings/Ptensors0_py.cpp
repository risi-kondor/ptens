

pybind11::class_<Ptensors0>(m,"ptensors0")

//.def(pybind11::init<const Ptensor0&, const Atoms&, const int, const FILLTYPE& dummy, const int _dev>(),"")

//.def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
//    return Ptensors0::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
//      return Ptensors0::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
//      return Ptensors0::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def("get_nc",&Ptensors0::get_nc)
  .def("atoms_of",&Ptensors0::atoms_of)
  .def("push_back",&Ptensors0::push_back)

  .def("str",&Ptensors0::str,py::arg("indent")="")
  .def("__str__",&Ptensors0::str,py::arg("indent")="");


