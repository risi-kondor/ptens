typedef SubgraphLayer0<Ptensors0> SGlayer0;
typedef SubgraphLayer1<Ptensors1> SGlayer1;
typedef SubgraphLayer2<Ptensors2> SGlayer2;


pybind11::class_<SGlayer1>(m,"subgraph_layer1")

//.def(pybind11::init<ptens::Ggraph&, const at::Tensor&>())

  .def_static("dummy",[]() {return SGlayer1();})

  .def_static("raw",[](const Ggraph& G, const Subgraph& S, const vector<vector<int> >& v, const int _nc, const int _dev){
      return SGlayer1(G,S,v,_nc,cnine::fill_raw(),_dev);}, py::arg("graph"),py::arg("subgraph"),py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const Ggraph& G, const Subgraph& S, const vector<vector<int> >& v, const int _nc, const int _dev){
      return SGlayer1(G,S,v,_nc,cnine::fill_zero(),_dev);}, py::arg("graph"),py::arg("subgraph"),py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const Ggraph& G, const Subgraph& S, const vector<vector<int> >& v, const int _nc, const int _dev){
      return SGlayer1(G,S,v,_nc,cnine::fill_gaussian(),_dev);}, py::arg("graph"),py::arg("subgraph"),py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const Ggraph& G, const Subgraph& S, const vector<vector<int> >& v, const int _nc, const int _dev){
      return SGlayer1(G,S,v,_nc,cnine::fill_sequential(),_dev);}, py::arg("graph"),py::arg("subgraph"),py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("zeros_like",&SGlayer1::zeros_like)
  .def_static("randn_like",&SGlayer1::randn_like)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("get_grad",&SGlayer1::get_grad)


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&SGlayer1::get_dev)
  .def("get_nc",&SGlayer1::get_nc)
  .def("get_atoms",[](const SGlayer1& x){return x.atoms.as_vecs();})
  .def("view_of_atoms",&SGlayer1::view_of_atoms)

  .def("atoms_of",[](const SGlayer1& x, const int i){return vector<int>(x.atoms_of(i));})
  .def("push_back",&SGlayer1::push_back)

  .def("to_device",&SGlayer1::to_device)
  .def("move_to_device_back",[](SGlayer1& x, SGlayer1& g, const int dev){
      if(!x.grad) x.grad=new SGlayer1(g.get_grad(),dev);
      else x.grad->add(SGlayer1(g.get_grad(),dev));})


// ---- Operations -------------------------------------------------------------------------------------------


  .def(pybind11::init<const SGlayer0&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, SGlayer0& x){r.gather_back(x);})

  .def(pybind11::init<const SGlayer1&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, SGlayer1& x){r.gather_back(x);})

  .def(pybind11::init<const SGlayer2&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, SGlayer2& x){r.gather_back(x);})


  .def("add",[](SGlayer1& x, const SGlayer1& y){x.add(y);})

  .def("add_concat_back",[](SGlayer1& x, SGlayer1& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def("add_mprod",[](SGlayer1& r, const SGlayer1& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](SGlayer1& x, SGlayer1& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g.get_grad(),RtensorA::view(M));})
  .def("mprod_back1",[](SGlayer1& x, SGlayer1& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({x.nc,g.nc},g.dev);
      g.add_mprod_back1_to(R,x);
      return R.torch();})

/*
  .def("add_scale",[](SGlayer1& r, const SGlayer1& x, at::Tensor& y){
      RtensorA Y(y);
      Y.move_to_device(0);
      PTENS_ASSRT(Y.ndims()==1);
      PTENS_ASSRT(Y.dims[0]==1);
      r.add(x,Y(0));})
  .def("add_scale_back0",[](SGlayer1& x, const cnine::loose_ptr<SGlayer1>& g, at::Tensor& y){
      RtensorA Y(y);
      Y.move_to_device(0);
      PTENS_ASSRT(Y.ndims()==1);
      PTENS_ASSRT(Y.dims[0]==1);
      x.get_grad().add(g,Y(0));})
  .def("scale_back1",[](SGlayer1&x, const cnine::loose_ptr<SGlayer1>& g){
      RtensorA R(Gdims(1));
      R.set(0,x.inp(*g));
      return R.move_to_device(g->dev).torch();})
  
  .def("scale_channels",[](SGlayer1& x, at::Tensor& y){
      return x.scale_channels(RtensorA::view(y).view1());})
  .def("add_scale_channels",[](SGlayer1& r, const SGlayer1& x, at::Tensor& y){
      r.add_scale_channels(x,RtensorA::view(y).view1());})
  .def("add_scale_channels_back0",[](SGlayer1& r, const cnine::loose_ptr<SGlayer1>& g, at::Tensor& y){
      r.get_grad().add_scale_channels(g,RtensorA::view(y).view1());}) // changed 
*/

  .def("add_linear",[](SGlayer1& r, const SGlayer1& x, at::Tensor& y, at::Tensor& b){
      r.add_linear(x,RtensorA::view(y),RtensorA::view(b));})
  .def("add_linear_back0",[](SGlayer1& x, SGlayer1& g, at::Tensor& y){
      x.get_grad().add_mprod_back0(g.get_grad(),RtensorA::view(y));})
  .def("linear_back1",[](SGlayer1& x, SGlayer1& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({x.nc,g.nc},g.dev);
      g.add_linear_back1_to(R,x);
      return R.torch();})
  .def("linear_back2",[](SGlayer1& x, SGlayer1& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({g.nc},g.dev);
      g.add_linear_back2_to(R);
      return R.torch();})

  .def("add_ReLU",[](SGlayer1& r, const SGlayer1& x, const float alpha){
      r.add_ReLU(x,alpha);})

  .def("inp",[](const SGlayer1& x, const SGlayer1& y){return x.inp(y);})
  .def("diff2",[](const SGlayer1& x, const SGlayer1& y){return x.diff2(y);})


// ---- I/O --------------------------------------------------------------------------------------------------

  .def("str",&SGlayer1::str,py::arg("indent")="")
  .def("__str__",&SGlayer1::str,py::arg("indent")="")
  .def("__repr__",&SGlayer1::str,py::arg("indent")="");
