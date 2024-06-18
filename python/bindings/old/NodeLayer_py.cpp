typedef SubgraphLayer0<Ptensors0> SGlayer0;
typedef SubgraphLayer1<Ptensors1> SGlayer1;
typedef SubgraphLayer2<Ptensors2> SGlayer2;


pybind11::class_<NodeLayer>(m,"nodelayer")


//.def(pybind11::init<ptens::Ggraph&, const at::Tensor&>())

  .def_static("raw",[](const Ggraph& G, const int _nc, const int _dev){
      return NodeLayer(G,_nc,cnine::fill_raw(),_dev);}, py::arg("graph"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const Ggraph& G,const int _nc, const int _dev){
      return NodeLayer(G,_nc,cnine::fill_zero(),_dev);}, py::arg("graph"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const Ggraph& G,const int _nc, const int _dev){
      return NodeLayer(G,_nc,cnine::fill_gaussian(),_dev);}, py::arg("graph"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const Ggraph& G, const int _nc, const int _dev){
    return NodeLayer(G,_nc,cnine::fill_sequential(),_dev);}, py::arg("graph"),py::arg("nc"),py::arg("device")=0)

//.def_static("zeros_like",&NodeLayer::zeros_like)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


//.def("get_grad",&NodeLayer::get_grad)
//.def("get_gradp",&NodeLayer::get_gradp)
//.def("gradp",&NodeLayer::get_gradp)
//.def("add_to_grad",[](NodeLayer& x, const cnine::loose_ptr<Ptensors0>& y){x.add_to_grad(y);})

//.def("ptensors0",[](const NodeLayer& x){return Ptensors0(x);})
//.def("toPtensors0_back",[](NodeLayer& x, Ptensors0& r){
//    if(!x.grad) x.grad=new Ptensors0(r.get_grad());
//    else x.grad->add(r.get_grad());})


  .def("torch",[](const NodeLayer& x) {return x.torch();})
  .def("torch_back",[](NodeLayer& x, const at::Tensor& g){
      x.add_to_grad(NodeLayer(x.G,Tensor<float>(g)));})

  .def("to_device",[](NodeLayer& x, const int dev){return NodeLayer(x,dev);})
  .def("to_device_back",[](NodeLayer& x, NodeLayer& g, const int dev){
      if(!x.grad) x.add_to_grad(NodeLayer(g.get_grad(),dev));})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&NodeLayer::get_dev)
  .def("get_nc",&NodeLayer::get_nc)


// ---- Operations -------------------------------------------------------------------------------------------


  .def(pybind11::init<const SGlayer0&>())
  .def("gather_back",[](NodeLayer& r, SGlayer0& x){r.gather_back(x);})

  .def(pybind11::init<const SGlayer1&>())
  .def("gather_back",[](NodeLayer& r, SGlayer1& x){r.gather_back(x);})


  .def(pybind11::init<const Ggraph&, const Ptensors0&>())
  .def("gather_back",[](NodeLayer& r, Ptensors0& x){r.gather_back(x);})

  .def(pybind11::init<const Ggraph&, const Ptensors1&>())
  .def("gather_back",[](NodeLayer& r, Ptensors1& x){r.gather_back(x);})



  .def("add",[](NodeLayer& x, const NodeLayer& y){x.add(y);})
  .def("plus",[](const NodeLayer& x, const NodeLayer& y){
      NodeLayer r(x); r.add(y); return r;})

/*
  .def("concat",[](NodeLayer& x, NodeLayer& y){
      auto r=x.zeros(x.get_nc()+y.get_nc());
      r.add_to_channels(x,0);
      r.add_to_channels(y,x.get_nc());
      return r;
    })
  .def("add_concat_back",[](NodeLayer& x, NodeLayer& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def("mprod",[](const NodeLayer& x, at::Tensor& y){
      NodeLayer r(x.G,x.S,x.atoms,y.size(1),x.dev);
      r.add_mprod(x,RtensorA::view(y));
      return r;})
  .def("add_mprod",[](NodeLayer& r, const NodeLayer& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](NodeLayer& x, NodeLayer& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g.get_grad(),RtensorA::view(M));})
  .def("mprod_back1",[](NodeLayer& x, NodeLayer& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({x.nc,g.nc},g.dev);
      g.add_mprod_back1_to(R,x);
      return R.torch();})      

  .def("linear",[](const NodeLayer& x, at::Tensor& y, at::Tensor& b){
      NodeLayer r(x.G,x.S,x.atoms,y.size(1),x.dev);
      r.add_linear(x,RtensorA::view(y),RtensorA::view(b));
      return r;})
  .def("add_linear_back0",[](NodeLayer& x, NodeLayer& g, at::Tensor& y){
      x.get_grad().add_mprod_back0(g.get_grad(),RtensorA::view(y));})
  .def("linear_back1",[](NodeLayer& x, NodeLayer& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({x.nc,g.nc},g.dev);
      g.add_linear_back1_to(R,x);
      return R.torch();})
  .def("linear_back2",[](NodeLayer& x, NodeLayer& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({g.nc},g.dev);
      g.add_linear_back2_to(R);
      return R.torch();})

  .def("normalize_channels",[](NodeLayer& x){
      x.norms=x.inv_channel_norms();
      auto R=NodeLayer::zeros_like(x);
      R.add_scale_channels(x,x.norms.view1());
      return R;
    })
  .def("normalize_channels_back",[](NodeLayer& x, NodeLayer& g){
      x.get_grad().add_scale_channels(g.get_grad(),x.norms.view1());
    })
*/

  .def("inp",[](const NodeLayer& x, const NodeLayer& y){return x.inp(y);})
  .def("diff2",[](const NodeLayer& x, const NodeLayer& y){return x.diff2(y);})

/*
  .def("add_ReLU",[](NodeLayer& r, const NodeLayer& x, const float alpha){
      r.add_ReLU(x,alpha);})
  .def("add_ReLU_back",[](NodeLayer& x, NodeLayer& r, const float alpha){
      x.get_grad().add_ReLU_back(r.get_grad(),x,alpha);})
*/

// ---- I/O --------------------------------------------------------------------------------------------------

  .def("str",&NodeLayer::str,py::arg("indent")="")
  .def("__str__",&NodeLayer::str,py::arg("indent")="")
  .def("__repr__",&NodeLayer::str,py::arg("indent")="");

