pybind11::class_<Ptensors2/*,cnine::RtensorPack*/>(m,"ptensors2")

  .def(pybind11::init<const Ptensors2&>())
  .def(pybind11::init<const Ptensors2&, const int>())

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

  .def_static("cat",&Ptensors2::cat)
  .def("add_cat_back",[](Ptensors2& x, Ptensors2& r, const int offs){
      x.get_grad().add_subpack(r.get_grad(),offs);})

  .def_static("sum",&Ptensors2::cat)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](Ptensors2& x, const Ptensors2& y){x.add_to_grad(y);})
  .def("add_to_grad",[](Ptensors2& x, const Ptensors2& y, const float c){x.add_to_grad(y,c);})
  .def("add_to_grad",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& y){x.add_to_grad(y);})
  .def("get_grad",[](Ptensors2& x){return x.get_grad();})
  .def("get_gradp",&Ptensors2::get_gradp)
  .def("gradp",&Ptensors2::get_gradp)
//  .def("view_of_grad",&Ptensors2::view_of_grad)

  .def("add_to_grad",[](Ptensors2& x, const int i, at::Tensor& T){
      x.get_grad().view_of_tensor(i).add(RtensorA::view(T));})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&Ptensors2::get_dev)
  .def("__len__",&Ptensors2::size)
  .def("get_nc",&Ptensors2::get_nc)
  .def("get_atoms",[](const Ptensors2& x){return x.atoms.as_vecs();})
//  .def("atoms",&Ptensors2::atoms)
//.def("view_of_atoms",&Ptensors2::view_of_atoms)
  .def("view_of_atoms",[](const Ptensors2& x){return x.atoms;})

  .def("__getitem__",[](const Ptensors2& x, const int i){return x(i);})
  .def("torch",[](const Ptensors2& x){return x.tensor().torch();})

  .def("atoms_of",[](const Ptensors2& x, const int i){return vector<int>(x.atoms_of(i));})
  .def("push_back",&Ptensors2::push_back)

  .def("to_device",&Ptensors2::to_device)
  .def("move_to_device_back",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g, const int dev){
      if(!x.grad) x.grad=new Ptensors2(g,dev);
      else x.grad->add(Ptensors2(g,dev));})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](Ptensors2& x, const Ptensors2& y){x.add(y);})

  .def("add_concat_back",[](Ptensors2& x, Ptensors2& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def("add_mprod",[](Ptensors2& r, const Ptensors2& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g,RtensorA::view(M));})
  .def("mprod_back1",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g){
      RtensorA R=RtensorA::zero({x.nc,g->nc},g->dev);
      g->add_mprod_back1_to(R,x);
      return R.torch();
    })

  .def("add_scale",[](Ptensors2& r, const Ptensors2& x, at::Tensor& y){
      RtensorA Y(y);
      Y.move_to_device(0);
      PTENS_ASSRT(Y.ndims()==1);
      PTENS_ASSRT(Y.dims[0]==1);
      r.add(x,Y(0));})
  .def("add_scale_back0",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g, at::Tensor& y){
      RtensorA Y(y);
      Y.move_to_device(0);
      PTENS_ASSRT(Y.ndims()==1);
      PTENS_ASSRT(Y.dims[0]==1);
      x.get_grad().add(g,Y(0));})
  .def("scale_back1",[](Ptensors2&x, const cnine::loose_ptr<Ptensors2>& g){
      RtensorA R(Gdims(1));
      R.set(0,x.inp(*g));
      return R.move_to_device(g->dev).torch();})
  
  .def("scale_channels",[](Ptensors2& x, at::Tensor& y){
      return x.scale_channels(RtensorA::view(y).view1());})
  .def("add_scale_channels",[](Ptensors2& r, const Ptensors2& x, at::Tensor& y){
      return r.add_scale_channels(x,RtensorA::view(y).view1());})
  .def("add_scale_channels_back0",[](Ptensors2& r, const cnine::loose_ptr<Ptensors2>& g, at::Tensor& y){
      r.get_grad().add_scale_channels(g,RtensorA::view(y).view1());})

  .def("add_linear",[](Ptensors2& r, const Ptensors2& x, at::Tensor& y, at::Tensor& b){
      r.add_linear(x,RtensorA::view(y),RtensorA::view(b));})
  .def("add_linear_back0",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g, at::Tensor& y){
      x.get_grad().add_mprod_back0(g,RtensorA::view(y));})
  .def("linear_back1",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g){
      RtensorA R=RtensorA::zero({x.nc,g->nc},g->dev);
      g->add_linear_back1_to(R,x);
      return R.torch();})
  .def("linear_back2",[](const Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g){
      RtensorA R=RtensorA::zero({g->nc},g->dev);
      g->add_linear_back2_to(R);
      return R.torch();})

  .def("add_ReLU",[](Ptensors2& r, const Ptensors2& x, const float alpha){
      r.add_ReLU(x,alpha);})
  .def("add_ReLU_back",[](Ptensors2& x, const cnine::loose_ptr<Ptensors2>& g, const float alpha){
      x.get_grad().add_ReLU_back(g,x,alpha);})

  .def("inp",[](const Ptensors2& x, const Ptensors2& y){return x.inp(y);})
  .def("diff2",[](const Ptensors2& x, const Ptensors2& y){return x.diff2(y);})

// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&Ptensors2::str,py::arg("indent")="")
  .def("__str__",&Ptensors2::str,py::arg("indent")="")
  .def("__repr__",&Ptensors2::repr);



pybind11::class_<loose_ptr<Ptensors2> >(m,"ptensors2_lptr");
