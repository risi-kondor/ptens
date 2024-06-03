typedef SubgraphLayer0<Ptensors0> SGlayer0;
typedef SubgraphLayer1<Ptensors1> SGlayer1;
typedef SubgraphLayer2<Ptensors2> SGlayer2;


pybind11::class_<SGlayer1,Ptensors1>(m,"subgraph_layer1")

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
//.def_static("randn_like",&SGlayer1::randn_like)

  .def("zeros",&SGlayer1::zeros_like)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("get_grad",[](SGlayer1& x){return x.get_grad();})
  .def("get_gradp",&Ptensors1::get_gradp)
  .def("gradp",&Ptensors1::get_gradp)
  .def("add_to_grad",[](SGlayer1& x, const Ptensors1& y){x.add_to_grad(y);})
  .def("add_to_grad",[](SGlayer1& x, const Ptensors1& y, const float c){x.add_to_grad(y,c);})
  .def("add_to_grad",[](SGlayer1& x, const cnine::loose_ptr<Ptensors1>& y){x.add_to_grad(y);})

  .def("ptensors1",[](const SGlayer1& x){return Ptensors1(x);})
  .def("toPtensors1_back",[](SGlayer1& x, Ptensors1& r){
      if(!x.grad) x.grad=new Ptensors1(r.get_grad());
      else x.grad->add(r.get_grad());})

  .def("torch",[](const SGlayer1& x){return x.tensor().torch();})
  .def("torch_back",[](SGlayer1& x, const at::Tensor& g){
      x.get_grad().add(Ptensors1(g,x.atoms));})
  .def_static("like",[](const SGlayer1& x, const at::Tensor& M){
      return SGlayer1::like(x,RtensorA(M));})

  .def("to_device",[](SGlayer1& x, const int dev){return SGlayer1(x,dev);})
  .def("to_device_back",[](SGlayer1& x, SGlayer1& g, const int dev){
      if(!x.grad) x.grad=new Ptensors1(g.get_grad(),dev);
      else x.grad->add(g.get_grad(),dev);})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&SGlayer1::get_dev)
  .def("get_nc",&SGlayer1::get_nc)
  .def("get_atoms",[](const SGlayer1& x){return x.atoms.as_vecs();})
  .def("view_of_atoms",&SGlayer1::view_of_atoms)

  .def("atoms_of",[](const SGlayer1& x, const int i){return vector<int>(x.atoms_of(i));})
  .def("push_back",&SGlayer1::push_back)


// ---- Operations -------------------------------------------------------------------------------------------


  .def(pybind11::init<const NodeLayer&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, NodeLayer& x){r.gather_back(x);})

  .def(pybind11::init<const SGlayer0&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, SGlayer0& x){r.gather_back(x);})

  .def(pybind11::init<const SGlayer1&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, SGlayer1& x){r.gather_back(x);})

  .def(pybind11::init<const SGlayer2&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, SGlayer2& x){r.gather_back(x);})


  .def(pybind11::init<const Ptensors0&, const Ggraph&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, Ptensors0& x){r.gather_back(x);})

  .def(pybind11::init<const Ptensors1&, const Ggraph&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, Ptensors1& x){r.gather_back(x);})

  .def(pybind11::init<const Ptensors2&, const Ggraph&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, Ptensors2& x){r.gather_back(x);})


  .def("add",[](SGlayer1& x, const SGlayer1& y){x.add(y);})
  .def("add",[](SGlayer1& x, const Ptensors1& y){x.Ptensors1::add(y);})
  .def("plus",[](const SGlayer1& x, const SGlayer1& y){
      SGlayer1 r(x); r.add(y); return r;})

  .def("concat",[](SGlayer1& x, SGlayer1& y){
      auto r=x.zeros(x.get_nc()+y.get_nc());
      r.add_to_channels(x,0);
      r.add_to_channels(y,x.get_nc());
      return r;
    })
  .def("add_concat_back",[](SGlayer1& x, SGlayer1& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def("mprod",[](const SGlayer1& x, at::Tensor& y){
      SGlayer1 r(x.G,x.S,x.atoms,y.size(1),x.dev);
      r.add_mprod(x,RtensorA::view(y));
      return r;})
  .def("add_mprod",[](SGlayer1& r, const SGlayer1& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](SGlayer1& x, SGlayer1& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g.get_grad(),RtensorA::view(M));})
  .def("mprod_back1",[](SGlayer1& x, SGlayer1& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({x.nc,g.nc},g.dev);
      g.add_mprod_back1_to(R,x);
      return R.torch();})

  .def("linear",[](const SGlayer1& x, at::Tensor& y, at::Tensor& b){
      SGlayer1 r(x.G,x.S,x.atoms,y.size(1),x.dev);
      r.add_linear(x,RtensorA::view(y),RtensorA::view(b));
      return r;})
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

  .def("autobahn",[](const SGlayer1& x, at::Tensor& W, at::Tensor& B){
      return x.autobahn(RtensorA::view(W),RtensorA::view(B));})
  .def("add_autobahn_back0",[](SGlayer1& x, SGlayer1& r, at::Tensor& W){
      x.add_autobahn_back0(r,RtensorA::view(W));})
  .def("autobahn_back1",[](SGlayer1& x, at::Tensor& W, at::Tensor& B, SGlayer1& r){
      x.add_autobahn_back1_to(RtensorA::view(W), RtensorA::view(B),r);})

  .def("inp",[](const SGlayer1& x, const SGlayer1& y){return x.inp(y);})
  .def("diff2",[](const SGlayer1& x, const SGlayer1& y){return x.diff2(y);})

  .def("ReLU",[](const SGlayer1& x, const float alpha){
      auto r=x.zeros(); r.add_ReLU(x,alpha); return r;})
  .def("add_ReLU_back",[](SGlayer1& x, SGlayer1& r, const float alpha){
      x.get_grad().add_ReLU_back(r.get_grad(),x,alpha);})


// ---- I/O --------------------------------------------------------------------------------------------------

  .def("str",&SGlayer1::str,py::arg("indent")="")
  .def("__str__",&SGlayer1::str,py::arg("indent")="")
  .def("__repr__",&SGlayer1::repr);

