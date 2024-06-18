pybind11::class_<Ptensors1/*,cnine::RtensorPack*/>(m,"ptensors1")

  .def(pybind11::init<const Ptensors1&>())
  .def(pybind11::init<const Ptensors1&, const int>())

  .def(pybind11::init<const at::Tensor&, const AtomsPack&>())
  .def(pybind11::init<const at::Tensor&, const vector<vector<int> >& >())

  .def_static("dummy",[]() {return Ptensors1(0,0);})
  
  .def_static("raw",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors1::raw(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors1::zero(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<vector<int> >& v, const int _nc, const float sigma, const int _dev){
      return Ptensors1::gaussian(AtomsPack(v),_nc,sigma,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("sigma"),py::arg("device")=0)
  .def_static("sequential",[](const vector<vector<int> >& v, const int _nc, const int _dev){
      return Ptensors1::sequential(AtomsPack(v),_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("raw",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors1::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const AtomsPack& _atoms, const int _nc, const int _dev){
      return Ptensors1::zero(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      return Ptensors1::gaussian(_atoms,_nc,sigma,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("sigma"),py::arg("device")=0)
  .def_static("sequential",[](const AtomsPack& _atoms, const int _nc, const int _dev){
    return Ptensors1::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("concat",&Ptensors1::concat)
  .def("add_concat_back",[](Ptensors1& x, Ptensors1& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def_static("cat",&Ptensors1::cat)
  .def("add_cat_back",[](Ptensors1& x, Ptensors1& r, const int offs){
      x.get_grad().add_subpack(r.get_grad(),offs);})

  .def_static("sum",&Ptensors1::cat)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](Ptensors1& x, const Ptensors1& y){x.add_to_grad(y);})
  .def("add_to_grad",[](Ptensors1& x, const Ptensors1& y, const float c){x.add_to_grad(y,c);})
  .def("add_to_grad",[](Ptensors1& x, const cnine::loose_ptr<Ptensors1>& y){x.add_to_grad(y);})
  .def("get_grad",[](Ptensors1& x){return x.get_grad();})
  .def("get_gradp",&Ptensors1::get_gradp)
  .def("gradp",&Ptensors1::get_gradp)
  .def("add_to_grad",[](Ptensors1& x, const int i, at::Tensor& T){
      x.get_grad().view_of_tensor(i).add(RtensorA::view(T));})



// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&Ptensors1::get_dev)
  .def("__len__",&Ptensors1::size)
  .def("get_nc",&Ptensors1::get_nc)
  .def("get_atoms",[](const Ptensors1& x){return x.atoms.as_vecs();})
  .def("view_of_atoms",[](const Ptensors1& x){return x.atoms;})
//.def("atoms",&Ptensors1::atoms)
//.def("get_atomsref",&Ptensors1::get_atomsref)
//.def("view_of_atoms",&Ptensors1::view_of_atoms)

  .def("__getitem__",[](const Ptensors1& x, const int i){return x(i);})

  .def("torch",[](const Ptensors1& x){return x.tensor().torch();})
  .def_static("like",[](const Ptensors1& x, const at::Tensor& M){
      return Ptensors1::like(x,RtensorA(M));})

  .def("atoms_of",[](const Ptensors1& x, const int i){return vector<int>(x.atoms_of(i));})
  .def("push_back",&Ptensors1::push_back)

  .def("to_device",&Ptensors1::to_device)
  .def("move_to_device",&Ptensors1::to_device)
  .def("move_to_device_back",[](Ptensors1& x, const cnine::loose_ptr<Ptensors1>& g, const int dev){
      if(!x.grad) x.grad=new Ptensors1(g,dev);
      else x.grad->add(Ptensors1(g,dev));})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](Ptensors1& x, const Ptensors1& y){x.add(y);})

  .def("add_mprod",[](Ptensors1& r, const Ptensors1& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](Ptensors1& x, const cnine::loose_ptr<Ptensors1>& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g,RtensorA::view(M));})
  .def("mprod_back1",[](Ptensors1& x, const cnine::loose_ptr<Ptensors1>& g){
      RtensorA R=RtensorA::zero({x.nc,g->nc},g->dev);
      g->add_mprod_back1_to(R,x);
      return R.torch();
    })

  .def("add_scale",[](Ptensors1& r, const Ptensors1& x, at::Tensor& y){
      RtensorA Y(y);
      Y.move_to_device(0);
      PTENS_ASSRT(Y.ndims()==1);
      PTENS_ASSRT(Y.dims[0]==1);
      r.add(x,Y(0));})
  .def("add_scale_back0",[](Ptensors1& x, const cnine::loose_ptr<Ptensors1>& g, at::Tensor& y){
      RtensorA Y(y);
      Y.move_to_device(0);
      PTENS_ASSRT(Y.ndims()==1);
      PTENS_ASSRT(Y.dims[0]==1);
      x.get_grad().add(g,Y(0));})
  .def("scale_back1",[](Ptensors1&x, const cnine::loose_ptr<Ptensors1>& g){
      RtensorA R(Gdims(1));
      R.set(0,x.inp(*g));
      return R.move_to_device(g->dev).torch();})
  
  .def("scale_channels",[](Ptensors1& x, at::Tensor& y){
      return x.scale_channels(RtensorA::view(y).view1());})
  .def("add_scale_channels",[](Ptensors1& r, const Ptensors1& x, at::Tensor& y){
      return r.add_scale_channels(x,RtensorA::view(y).view1());})
  .def("add_scale_channels_back0",[](Ptensors1& r, const cnine::loose_ptr<Ptensors1>& g, at::Tensor& y){
      r.get_grad().add_scale_channels(g,RtensorA::view(y).view1());})

  .def("add_linear",[](Ptensors1& r, const Ptensors1& x, at::Tensor& y, at::Tensor& b){
      r.add_linear(x,RtensorA::view(y),RtensorA::view(b));})
  .def("add_linear_back0",[](Ptensors1& x, const cnine::loose_ptr<Ptensors1>& g, at::Tensor& y){
      x.get_grad().add_mprod_back0(g,RtensorA::view(y));})
  .def("linear_back1",[](Ptensors1& x, const cnine::loose_ptr<Ptensors1>& g){
      RtensorA R=RtensorA::zero({x.nc,g->nc},g->dev);
      g->add_linear_back1_to(R,x);
      return R.torch();})
  .def("linear_back2",[](const Ptensors1& x, const cnine::loose_ptr<Ptensors1>& g){
      RtensorA R=RtensorA::zero({g->nc},g->dev);
      g->add_linear_back2_to(R);
      return R.torch();})

  .def("add_ReLU",[](Ptensors1& r, const Ptensors1& x, const float alpha){
      r.add_ReLU(x,alpha);})
  .def("add_ReLU_back",[](Ptensors1& x, const cnine::loose_ptr<Ptensors1>& g, const float alpha){
      x.get_grad().add_ReLU_back(g,x,alpha);})

  .def("inp",[](const Ptensors1& x, const Ptensors1& y){return x.inp(y);})
  .def("diff2",[](const Ptensors1& x, const Ptensors1& y){return x.diff2(y);})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&Ptensors1::str,py::arg("indent")="")
  .def("__str__",&Ptensors1::str,py::arg("indent")="")
  .def("__repr__",&Ptensors1::repr);




pybind11::class_<loose_ptr<Ptensors1> >(m,"ptensors1_lptr");

