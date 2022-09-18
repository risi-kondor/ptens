

pybind11::class_<Ptensors1>(m,"ptensors1")

//.def(pybind11::init<const Ptensor0&, const Atoms&, const int, const FILLTYPE& dummy, const int _dev>(),"")

  .def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors1::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors1::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//.def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
//    return Ptensors1::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def("get_nc",&Ptensors1::get_nc)
  .def("atoms_of",&Ptensors1::atoms_of)
  .def("push_back",&Ptensors1::push_back)

  .def("str",&Ptensors1::str,py::arg("indent")="")
  .def("__str__",&Ptensors1::str,py::arg("indent")="");


