typedef SubgraphLayer0<Ptensors0> SGlayer0;
typedef SubgraphLayer1<Ptensors1> SGlayer1;
typedef SubgraphLayer2<Ptensors2> SGlayer2;


pybind11::class_<SGlayer2,Ptensors2>(m,"subgraph_layer2")

  .def_static("dummy",[]() {return SGlayer2();})

  .def_static("raw",[](const Ggraph& G, const Subgraph& S, const vector<vector<int> >& v, const int _nc, const int _dev){
      return SGlayer2(G,S,v,_nc,cnine::fill_raw(),_dev);}, py::arg("graph"),py::arg("subgraph"),py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("zero",[](const Ggraph& G, const Subgraph& S, const vector<vector<int> >& v, const int _nc, const int _dev){
      return SGlayer2(G,S,v,_nc,cnine::fill_zero(),_dev);}, py::arg("graph"),py::arg("subgraph"),py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("gaussian",[](const Ggraph& G, const Subgraph& S, const vector<vector<int> >& v, const int _nc, const int _dev){
      return SGlayer2(G,S,v,_nc,cnine::fill_gaussian(),_dev);}, py::arg("graph"),py::arg("subgraph"),py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const Ggraph& G, const Subgraph& S, const vector<vector<int> >& v, const int _nc, const int _dev){
      return SGlayer2(G,S,v,_nc,cnine::fill_sequential(),_dev);}, py::arg("graph"),py::arg("subgraph"),py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("zeros_like",&SGlayer2::zeros_like)
  .def_static("randn_like",&SGlayer2::randn_like)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("get_grad",[](SGlayer2& x){return x.get_grad();})
  .def("get_gradp",&Ptensors2::get_gradp)
  .def("gradp",&Ptensors2::get_gradp)
  .def("add_to_grad",[](SGlayer2& x, const cnine::loose_ptr<Ptensors2>& y){x.add_to_grad(y);})

  .def("ptensors2",[](const SGlayer2& x){return Ptensors2(x);})
  .def("toPtensors2_back",[](SGlayer2& x, Ptensors2& r){
      if(!x.grad) x.grad=new Ptensors2(r.get_grad());
      else x.grad->add(r.get_grad());})

  .def("torch",[](const SGlayer2& x){return x.tensor().torch();})
  .def("torch_back",[](SGlayer2& x, const at::Tensor& g){
      x.get_grad().add(Ptensors2(g,x.atoms));})

  .def("to_device",[](SGlayer2& x, const int dev){return SGlayer2(x,dev);})
  .def("move_to_device_back",[](SGlayer2& x, SGlayer2& g, const int dev){
      if(!x.grad) x.grad=new Ptensors2(g.get_grad(),dev);
      else x.grad->add(g.get_grad(),dev);})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&SGlayer2::get_dev)
  .def("get_nc",&SGlayer2::get_nc)
  .def("get_atoms",[](const SGlayer2& x){return x.atoms.as_vecs();})
  .def("view_of_atoms",&SGlayer2::view_of_atoms)


  .def("atoms_of",[](const SGlayer2& x, const int i){return vector<int>(x.atoms_of(i));})


// ---- Operations -------------------------------------------------------------------------------------------


  .def(pybind11::init<const SGlayer0&, const Subgraph&>())
  .def("gather_back",[](SGlayer2& r, SGlayer0& x){r.gather_back(x);})

  .def(pybind11::init<const SGlayer1&, const Subgraph&>())
  .def("gather_back",[](SGlayer2& r, SGlayer1& x){r.gather_back(x);})

  .def(pybind11::init<const SGlayer2&, const Subgraph&>())
  .def("gather_back",[](SGlayer2& r, SGlayer2& x){r.gather_back(x);})

  .def(pybind11::init<const Ptensors0&, const Ggraph&, const Subgraph&>())
  .def("gather_back",[](SGlayer2& r, Ptensors0& x){r.gather_back(x);})

  .def(pybind11::init<const Ptensors1&, const Ggraph&, const Subgraph&>())
  .def("gather_back",[](SGlayer2& r, Ptensors1& x){r.gather_back(x);})

  .def(pybind11::init<const Ptensors2&, const Ggraph&, const Subgraph&>())
  .def("gather_back",[](SGlayer1& r, Ptensors2& x){r.gather_back(x);})


  .def("add",[](SGlayer2& x, const SGlayer2& y){x.add(y);})
  .def("plus",[](const SGlayer2& x, const SGlayer2& y){
      SGlayer2 r(x); r.add(y); return r;})

  .def("concat",[](SGlayer2& x, SGlayer2& y){
      auto r=x.zeros(x.get_nc()+y.get_nc());
      r.add_to_channels(x,0);
      r.add_to_channels(y,x.get_nc());
      return r;
    })
  .def("add_concat_back",[](SGlayer2& x, SGlayer2& g, const int offs){
      x.get_grad().add_channels(g.get_grad(),offs);})

  .def("mprod",[](const SGlayer2& x, at::Tensor& y){
      SGlayer2 r(x.G,x.S,x.atoms,y.size(1),x.dev);
      r.add_mprod(x,RtensorA::view(y));
      return r;})
  .def("add_mprod",[](SGlayer2& r, const SGlayer2& x, at::Tensor& y){
      r.add_mprod(x,RtensorA::view(y));})
  .def("add_mprod_back0",[](SGlayer2& x, SGlayer2& g, at::Tensor& M){
      x.get_grad().add_mprod_back0(g.get_grad(),RtensorA::view(M));})
  .def("mprod_back1",[](SGlayer2& x, SGlayer2& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({x.nc,g.nc},g.dev);
      g.add_mprod_back1_to(R,x);
      return R.torch();})

  .def("linear",[](const SGlayer2& x, at::Tensor& y, at::Tensor& b){
      SGlayer2 r(x.G,x.S,x.atoms,y.size(1),x.dev);
      r.add_linear(x,RtensorA::view(y),RtensorA::view(b));
      return r;})
  .def("add_linear",[](SGlayer2& r, const SGlayer2& x, at::Tensor& y, at::Tensor& b){
      r.add_linear(x,RtensorA::view(y),RtensorA::view(b));})
  .def("add_linear_back0",[](SGlayer2& x, SGlayer2& g, at::Tensor& y){
      x.get_grad().add_mprod_back0(g.get_grad(),RtensorA::view(y));})
  .def("linear_back1",[](SGlayer2& x, SGlayer2& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({x.nc,g.nc},g.dev);
      g.add_linear_back1_to(R,x);
      return R.torch();})
  .def("linear_back2",[](SGlayer2& x, SGlayer2& _g){
      auto& g=_g.get_grad();
      RtensorA R=RtensorA::zero({g.nc},g.dev);
      g.add_linear_back2_to(R);
      return R.torch();})

  .def("inp",[](const SGlayer2& x, const SGlayer2& y){return x.inp(y);})
  .def("diff2",[](const SGlayer2& x, const SGlayer2& y){return x.diff2(y);})

  .def("add_ReLU",[](SGlayer2& r, const SGlayer2& x, const float alpha){
      r.add_ReLU(x,alpha);})
  .def("add_ReLU_back",[](SGlayer2& x, SGlayer2& r, const float alpha){
      x.get_grad().add_ReLU_back(r.get_grad(),x,alpha);})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&SGlayer2::str,py::arg("indent")="")
  .def("__str__",&SGlayer2::str,py::arg("indent")="")
  .def("__repr__",&SGlayer2::repr);



