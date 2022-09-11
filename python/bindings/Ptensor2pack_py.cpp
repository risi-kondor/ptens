

pybind11::class_<Ptensor2pack>(m,"ptensor2pack")

//.def(pybind11::init<const Ptensor0&, const Atoms&, const int, const FILLTYPE& dummy, const int _dev>(),"")

  .def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensor2pack::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensor2pack::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
//      return Ptensor2pack::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def("get_nc",&Ptensor2pack::get_nc)
  .def("atoms_of",&Ptensor2pack::atoms_of)
  .def("push_back",&Ptensor2pack::push_back)

  .def("str",&Ptensor2pack::str,py::arg("indent")="")
  .def("__str__",&Ptensor2pack::str,py::arg("indent")="");


