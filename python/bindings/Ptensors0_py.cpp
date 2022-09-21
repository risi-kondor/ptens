

pybind11::class_<Ptensors0>(m,"ptensors0")

//.def(pybind11::init<const Ptensor0&, const Atoms&, const int, const FILLTYPE& dummy, const int _dev>(),"")

  .def_static("raw",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors0::raw(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors0::zero(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors0::gaussian(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors0::sequential(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors0::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors0::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors0::gaussian(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors0::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


//.def("add_to_grad",&Ptensors0::add_to_grad)
  .def("get_grad",&Ptensors0::get_grad)
  .def("get_gradp",&Ptensors0::get_grad)
//  .def("view_of_grad",&Ptensors0::view_of_grad)

//.def("device",&Ptensors0::get_device)
//.def("to",&Ptensors0::to_device)
//.def("to_device",&Ptensors0::to_device)
//.def("move_to",[](Ptensors0& x, const int _dev){x.move_to_device(_dev);})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_nc",&Ptensors0::get_nc)
  .def("view_of_atoms",&Ptensors0::view_of_atoms)

  .def("atoms_of",&Ptensors0::atoms_of)
  .def("push_back",&Ptensors0::push_back)

  .def("str",&Ptensors0::str,py::arg("indent")="")
  .def("__str__",&Ptensors0::str,py::arg("indent")="")
  .def("__repr__",&Ptensors0::str,py::arg("indent")="");


