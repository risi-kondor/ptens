pybind11::class_<Ptensors2,cnine::RtensorPack>(m,"ptensors2")

  .def(pybind11::init<const Ptensors2&>())
  .def(pybind11::init<const at::Tensor&, const AtomsPack&>())
  .def(pybind11::init<const at::Tensor&, const vector<vector<int> >&>())

  .def_static("dummy",[]() {return Ptensors2(0,0);})

  .def_static("raw",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors2::raw(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors2::zero(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<vector<int> >& v, const int _nc, const float sigma, const int _dev){
      return Ptensors2::gaussian(AtomsPack(v),_nc,sigma,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("sigma"),py::arg("device")=0)
  .def_static("sequential",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors2::sequential(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors2::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors2::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      return Ptensors2::gaussian(_atoms,_nc,sigma,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("sigma"),py::arg("device")=0)
  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors2::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("concat",&Ptensors2::concat)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](Ptensors2& x, const Ptensors2& y){x.add_to_grad(y);})
  .def("add_to_grad",[](Ptensors2& x, const Ptensors2& y, const float c){x.add_to_grad(y,c);})
  .def("add_to_grad",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& y){x.add_to_grad(y);})
  .def("get_grad",&Ptensors2::get_grad)
  .def("get_gradp",&Ptensors2::get_gradp)
  .def("gradp",&Ptensors2::get_gradp)
//  .def("view_of_grad",&Ptensors2::view_of_grad)

  .def("add_to_grad",[](Ptensors2& x, const int i, at::Tensor& T){
      x.get_grad().view_of_tensor(i).add(RtensorA::view(T));})

//.def("device",&Ptensors2::get_device)
//.def("to",&Ptensors2::to_device)
//.def("to_device",&Ptensors2::to_device)
//.def("move_to",[](Ptensors2& x, const int _dev){x.move_to_device(_dev);})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&Ptensors2::get_dev)
  .def("get_nc",&Ptensors2::get_nc)
  .def("get_atoms",[](const Ptensors2& x){return x.atoms.as_vecs();})
  .def("view_of_atoms",&Ptensors2::view_of_atoms)

  .def("__getitem__",[](const Ptensors2& x, const int i){return x(i);})
  .def("torch",[](const Ptensors2& x){return x.tensor().torch();})

  .def("atoms_of",[](const Ptensors2& x, const int i){return vector<int>(x.atoms_of(i));})
  .def("push_back",&Ptensors2::push_back)

  .def("to_device",&Ptensors2::to_device)


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](Ptensors2& x, const Ptensors2& y){x.add(y);})

  .def("add_concat_back",[](Ptensors2& x, Ptensors2& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def("add_mprod",[](Ptensors2& r, const Ptensors2& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](Ptensors2& x, const Ptensors2& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g,RtensorA::view(M));})
  .def("mprod_back1",[](Ptensors2& x, const Ptensors2& g){
      RtensorA R=RtensorA::zero({x.nc,g.nc});
      g.add_mprod_back1_to(R,x);
      return R.torch();
    })

  .def("add_linear",[](Ptensors2& r, const Ptensors2& x, at::Tensor& y, at::Tensor& b){
      r.add_linear(x,RtensorA::view(y),RtensorA::view(b));})
  .def("add_linear_back0",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g, at::Tensor& y){
      x.get_grad().add_mprod_back0(g,RtensorA::view(y));})
  .def("linear_back1",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g){
      RtensorA R=RtensorA::zero({x.nc,g->nc});
      g->add_linear_back1_to(R,x);
      return R.torch();})
  .def("linear_back2",[](const cnine::loose_ptr<Ptensors2>& g){
      RtensorA R=RtensorA::zero({g->nc});
      g->add_linear_back2_to(R);
      return R.torch();})

  .def("add_ReLU",[](Ptensors2& r, const Ptensors2& x, const float alpha){
      r.add_ReLU(x,alpha);})
  .def("add_ReLU_back",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g, const float alpha){
      x.get_grad().add_ReLU_back(g,x,alpha);})

  .def("inp",&Ptensors2::inp)
  .def("diff2",&Ptensors2::diff2)


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&Ptensors2::str,py::arg("indent")="")
  .def("__str__",&Ptensors2::str,py::arg("indent")="")
  .def("__repr__",&Ptensors2::str,py::arg("indent")="");



pybind11::class_<loose_ptr<Ptensors2> >(m,"ptensors2_lptr");
