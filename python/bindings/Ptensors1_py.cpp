

pybind11::class_<Ptensors1>(m,"ptensors1")

//.def(pybind11::init<const Ptensor0&, const Atoms&, const int, const FILLTYPE& dummy, const int _dev>(),"")

  .def_static("raw",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors1::raw(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors1::zero(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors1::gaussian(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors1::sequential(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors1::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors1::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors1::gaussian(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
    return Ptensors1::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


//.def("add_to_grad",&Ptensors1::add_to_grad)
  .def("get_grad",&Ptensors1::get_grad)
  .def("get_gradp",&Ptensors1::get_grad)
//  .def("view_of_grad",&Ptensors1::view_of_grad)

//.def("device",&Ptensors1::get_device)
//.def("to",&Ptensors1::to_device)
//.def("to_device",&Ptensors1::to_device)
//.def("move_to",[](Ptensors1& x, const int _dev){x.move_to_device(_dev);})


// ---- Access ----------------------------------------------------------------------------------------------

//.def_readwrite("atoms",&Ptensors1::atoms)

  .def("get_nc",&Ptensors1::get_nc)
  .def("get_atomsref",&Ptensors1::get_atomsref)
  .def("atoms_of",&Ptensors1::atoms_of)
  .def("push_back",&Ptensors1::push_back)

  .def("str",&Ptensors1::str,py::arg("indent")="")
  .def("__str__",&Ptensors1::str,py::arg("indent")="")
  .def("__repr__",&Ptensors1::str,py::arg("indent")="");


