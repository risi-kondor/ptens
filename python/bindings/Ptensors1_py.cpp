pybind11::class_<Ptensors1,RtensorPool>(m,"ptensors1")

  .def(pybind11::init<const Ptensors1&>())

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

  .def_static("concat",&Ptensors1::concat)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](Ptensors1& x, const Ptensors1& y){x.add_to_grad(y);})
  .def("add_to_grad",[](Ptensors1& x, const loose_ptr<Ptensors1>& y){x.add_to_grad(y);})
  .def("get_grad",&Ptensors1::get_grad)
  .def("get_gradp",&Ptensors1::get_gradp)
  .def("gradp",&Ptensors1::get_gradp)
//  .def("view_of_grad",&Ptensors1::view_of_grad)

  .def("add_to_grad",[](Ptensors1& x, const int i, at::Tensor& T){
      x.get_grad().view_of_tensor(i).add(RtensorA::view(T));})

//.def("device",&Ptensors1::get_device)
//.def("to",&Ptensors1::to_device)
//.def("to_device",&Ptensors1::to_device)
//.def("move_to",[](Ptensors1& x, const int _dev){x.move_to_device(_dev);})


// ---- Access ----------------------------------------------------------------------------------------------

//.def_readwrite("atoms",&Ptensors1::atoms)

  .def("get_nc",&Ptensors1::get_nc)
  .def("get_atomsref",&Ptensors1::get_atomsref)
  .def("view_of_atoms",&Ptensors1::view_of_atoms)

  .def("__getitem__",[](const Ptensors1& x, const int i){return x(i);})

  .def("atoms_of",[](const Ptensors1& x, const int i){return vector<int>(x.atoms_of(i));})
  .def("push_back",&Ptensors1::push_back)


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",&Ptensors1::add)

  .def("add_concat_back",[](Ptensors1& x, Ptensors1& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def("add_mprod",[](Ptensors1& r, const Ptensors1& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](Ptensors1& x, const Ptensors1& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g,RtensorA::view(M));})
  .def("mprod_back1",[](Ptensors1& x, const Ptensors1& g){
      RtensorA R=RtensorA::zero({x.nc,g.nc});
      g.add_mprod_back1_to(R,x);
      return R.torch();
    })

// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&Ptensors1::str,py::arg("indent")="")
  .def("__str__",&Ptensors1::str,py::arg("indent")="")
  .def("__repr__",&Ptensors1::str,py::arg("indent")="");




pybind11::class_<loose_ptr<Ptensors1> >(m,"ptensors1_lptr");

