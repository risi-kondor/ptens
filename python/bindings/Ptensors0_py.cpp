pybind11::class_<RtensorPool>(m,"rtensor_pool");

pybind11::class_<Ptensors0,RtensorPool>(m,"ptensors0")

  .def(pybind11::init<const at::Tensor&>())
  .def(pybind11::init<const Ptensors0&>())

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

  .def_static("concat",&Ptensors0::concat)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](Ptensors0& x, const Ptensors0& y){x.add_to_grad(y);})
  .def("add_to_grad",[](Ptensors0& x, const loose_ptr<Ptensors0>& y){x.add_to_grad(y);})
//  .def("add_to_gradp",[](Ptensors0& x, const loose_ptr<Ptensors0>& y){x.add_to_grad(y);})
//.def("add_to_grad",&Ptensors0::add_to_grad)
  .def("get_grad",&Ptensors0::get_grad)
  .def("get_gradp",&Ptensors0::get_gradp)
  .def("gradp",&Ptensors0::get_gradp)

  .def("add_to_grad",[](Ptensors0& x, const int i, at::Tensor& T){
      x.get_grad().view_of_tensor(i).add(RtensorA::view(T));
    })

//  .def("view_of_grad",&Ptensors0::view_of_grad)

//.def("device",&Ptensors0::get_device)
//.def("to",&Ptensors0::to_device)
//.def("to_device",&Ptensors0::to_device)
//.def("move_to",[](Ptensors0& x, const int _dev){x.move_to_device(_dev);})

  .def("__getitem__",[](const Ptensors0& x, const int i){return x(i);})
  .def("torch",[](const Ptensors0& x){return x.tensor().torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_nc",&Ptensors0::get_nc)
  .def("view_of_atoms",&Ptensors0::view_of_atoms)


//.def("atoms_of",&Ptensors0::atoms_of)
  .def("atoms_of",[](const Ptensors0& x, const int i){return vector<int>(x.atoms_of(i));})
  .def("push_back",&Ptensors0::push_back)



// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",&Ptensors0::add)

  .def("add_concat_back",[](Ptensors0& x, Ptensors0& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def("add_mprod",[](Ptensors0& r, const Ptensors0& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](Ptensors0& x, const loose_ptr<Ptensors0>& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g,RtensorA::view(M));})
  .def("mprod_back1",[](Ptensors0& x, const loose_ptr<Ptensors0>& g){
      RtensorA R=RtensorA::zero({x.nc,g->nc});
      g->add_mprod_back1_to(R,x);
      return R.torch();
    })


// ---- I/O --------------------------------------------------------------------------------------------------

  .def("str",&Ptensors0::str,py::arg("indent")="")
  .def("__str__",&Ptensors0::str,py::arg("indent")="")
  .def("__repr__",&Ptensors0::str,py::arg("indent")="");


pybind11::class_<loose_ptr<Ptensors0> >(m,"ptensors0_lptr");
