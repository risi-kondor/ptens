

pybind11::class_<Ptensors2>(m,"ptensors2")

//.def(pybind11::init<const Ptensor0&, const Atoms&, const int, const FILLTYPE& dummy, const int _dev>(),"")

  .def_static("raw",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors2::raw(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors2::zero(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors2::gaussian(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors2::sequential(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors2::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors2::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors2::gaussian(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors2::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

// ---- Conversions, transport, etc. ------------------------------------------------------------------------


//.def("add_to_grad",&Ptensors2::add_to_grad)
  .def("get_grad",&Ptensors2::get_grad)
  .def("get_gradp",&Ptensors2::get_grad)
//  .def("view_of_grad",&Ptensors2::view_of_grad)

//.def("device",&Ptensors2::get_device)
//.def("to",&Ptensors2::to_device)
//.def("to_device",&Ptensors2::to_device)
//.def("move_to",[](Ptensors2& x, const int _dev){x.move_to_device(_dev);})


// ---- Access ----------------------------------------------------------------------------------------------

  .def("get_nc",&Ptensors2::get_nc)
  .def("view_of_atoms",&Ptensors2::view_of_atoms)

  .def("atoms_of",&Ptensors2::atoms_of)
  .def("push_back",&Ptensors2::push_back)

  .def("str",&Ptensors2::str,py::arg("indent")="")
  .def("__str__",&Ptensors2::str,py::arg("indent")="")
  .def("__repr__",&Ptensors2::str,py::arg("indent")="");


