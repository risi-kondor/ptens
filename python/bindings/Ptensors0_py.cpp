pybind11::class_<cnine::RtensorPack>(m,"rtensor_pack");

pybind11::class_<Ptensors0,cnine::RtensorPack>(m,"ptensors0")

  .def(pybind11::init<const at::Tensor&>())
  .def(pybind11::init<const Ptensors0&>())
  .def(pybind11::init<const Ptensors0&, const int>())

  .def_static("dummy",[]() {return Ptensors0(0,0);})

  .def_static("raw",[](const int n, const int _nc, const int _dev){
      return Ptensors0::raw(n,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const int n, const int _nc, const int _dev){
      return Ptensors0::zero(n,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const int n, const int _nc, const float sigma, const int _dev){
      return Ptensors0::gaussian(n,_nc,sigma,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("sigma"),py::arg("device")=0)
  .def_static("sequential",[](const int n, const int _nc, const int _dev){
      return Ptensors0::sequential(n,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("raw",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors0::raw(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors0::zero(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<vector<int> >& v, const int _nc, const float sigma, const int _dev){
      return Ptensors0::gaussian(AtomsPack(v),_nc,sigma,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("sigma"),py::arg("device")=0)
  .def_static("sequential",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors0::sequential(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors0::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors0::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      return Ptensors0::gaussian(_atoms,_nc,sigma,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("sigma"),py::arg("device")=0)
  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors0::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("concat",&Ptensors0::concat)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](Ptensors0& x, const Ptensors0& y){x.add_to_grad(y);})
  .def("add_to_grad",[](Ptensors0& x, const Ptensors0& y, const float c){x.add_to_grad(y,c);})
  .def("add_to_grad",[](Ptensors0& x, const cnine::loose_ptr<Ptensors0>& y){x.add_to_grad(y);})
//  .def("add_to_gradp",[](Ptensors0& x, const cnine::loose_ptr<Ptensors0>& y){x.add_to_grad(y);})
//.def("add_to_grad",&Ptensors0::add_to_grad)
  .def("get_grad",&Ptensors0::get_grad)
  .def("get_gradp",&Ptensors0::get_gradp)
  .def("gradp",&Ptensors0::get_gradp)

  .def("add_to_grad",[](Ptensors0& x, const int i, at::Tensor& T){
      x.get_grad().view_of_tensor(i).add(RtensorA::view(T));
    })

//  .def("view_of_grad",&Ptensors0::view_of_grad)

  .def("__getitem__",[](const Ptensors0& x, const int i){return x(i);})
  .def("torch",[](const Ptensors0& x){return x.tensor().torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&Ptensors0::get_dev)
  .def("get_nc",&Ptensors0::get_nc)
  .def("get_atoms",[](const Ptensors0& x){return x.atoms.as_vecs();})
  .def("view_of_atoms",&Ptensors0::view_of_atoms)


//.def("atoms_of",&Ptensors0::atoms_of)
  .def("atoms_of",[](const Ptensors0& x, const int i){return vector<int>(x.atoms_of(i));})
  .def("push_back",&Ptensors0::push_back)

  .def("to_device",&Ptensors0::to_device)
  .def("move_to_device_back",[](Ptensors0& x, const cnine::loose_ptr<Ptensors0>& g, const int dev){
      if(!x.grad) x.grad=new Ptensors0(g,dev);
      else x.grad->add(Ptensors0(g,dev));})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](Ptensors0& x, const Ptensors0& y){x.add(y);})

  .def("add_concat_back",[](Ptensors0& x, Ptensors0& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def("add_mprod",[](Ptensors0& r, const Ptensors0& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](Ptensors0& x, const cnine::loose_ptr<Ptensors0>& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g,RtensorA::view(M));})
  .def("mprod_back1",[](Ptensors0& x, const cnine::loose_ptr<Ptensors0>& g){
      RtensorA R=RtensorA::zero({x.nc,g->nc},g->dev);
      g->add_mprod_back1_to(R,x);
      return R.torch();})

  .def("scale_channels",[](Ptensors0& x, at::Tensor& y){
      return x.scale_channels(RtensorA::view(y).view1());})
  .def("add_scale_channels",[](Ptensors0& r, const Ptensors0& x, at::Tensor& y){
      r.add_scale_channels(x,RtensorA::view(y).view1());})
  .def("add_scale_channels_back0",[](Ptensors0& r, const cnine::loose_ptr<Ptensors0>& g, at::Tensor& y){
      r.get_grad().add_scale_channels(g,RtensorA::view(y).view1());}) // changed 

  .def("add_linear",[](Ptensors0& r, const Ptensors0& x, at::Tensor& y, at::Tensor& b){
      r.add_linear(x,RtensorA::view(y),RtensorA::view(b));})
  .def("add_linear_back0",[](Ptensors0& x, const cnine::loose_ptr<Ptensors0>& g, at::Tensor& y){
      x.get_grad().add_mprod_back0(g,RtensorA::view(y));})
  .def("linear_back1",[](Ptensors0& x, const cnine::loose_ptr<Ptensors0>& g){
      RtensorA R=RtensorA::zero({x.nc,g->nc},g->dev);
      g->add_linear_back1_to(R,x);
      return R.torch();})
  .def("linear_back2",[](Ptensors0& x, const cnine::loose_ptr<Ptensors0>& g){
      RtensorA R=RtensorA::zero({g->nc},g->dev);
      g->add_linear_back2_to(R);
      return R.torch();})

  

//.def("add_gather",[](Ptensors0& r const Ptensors0& x, const Hgraph& G){
//      r.add_gather(x,G);})
//  .def("add_gather_back",[](Ptensors0& r const Ptensors0& x, const Hgraph& G){
//      r.add_gather(x,G);})

  .def("add_ReLU",[](Ptensors0& r, const Ptensors0& x, const float alpha){
      r.add_ReLU(x,alpha);})
  .def("add_ReLU_back",[](Ptensors0& x, const cnine::loose_ptr<Ptensors0>& g, const float alpha){
      x.get_grad().add_ReLU_back(g,x,alpha);}) // forward is same as backward

  .def("inp",&Ptensors0::inp)
  .def("diff2",&Ptensors0::diff2)


// ---- I/O --------------------------------------------------------------------------------------------------

  .def("str",&Ptensors0::str,py::arg("indent")="")
  .def("__str__",&Ptensors0::str,py::arg("indent")="")
  .def("__repr__",&Ptensors0::str,py::arg("indent")="");


pybind11::class_<loose_ptr<Ptensors0> >(m,"ptensors0_lptr");



//.def("ReLU",[](const Ptensors0& x, const float alpha){
//      Ptensors0 R=Ptensors0::zeros_like(x);
//      R.add_ReLU(x,alpha);
//      return R;})
