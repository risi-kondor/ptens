

pybind11::class_<Ptensor0pack>(m,"ptensor0pack")

//.def(pybind11::init<const Ptensor0&, const Atoms&, const int, const FILLTYPE& dummy, const int _dev>(),"")

//.def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
//    return Ptensor0pack::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
//      return Ptensor0pack::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
//      return Ptensor0pack::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def("get_nc",&Ptensor0pack::get_nc)
  .def("atoms_of",&Ptensor0pack::atoms_of)
  .def("push_back",&Ptensor0pack::push_back)

  .def("str",&Ptensor0pack::str,py::arg("indent")="")
  .def("__str__",&Ptensor0pack::str,py::arg("indent")="");


