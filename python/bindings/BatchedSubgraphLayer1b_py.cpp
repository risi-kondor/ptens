typedef BatchedSubgraphLayer0b<float> BSGlayer0b;
typedef BatchedSubgraphLayer1b<float> BSGlayer1b;
typedef BatchedSubgraphLayer2b<float> BSGlayer2b;


pybind11::class_<BSGlayer1b,BatchedPtensors1b<float> >(m,"batched_subgraphlayer1b")

  .def_static("from_edge_features",[](const vector<int>& keys, at::Tensor&M){
      return BSGlayer1b::from_edge_features(keys,ATview<float>(M));})

  .def_static("like",[](const BSGlayer1b& x, at::Tensor& M){
      return BSGlayer1b(x.G,x.S,x.atoms,ATview<float>(M));})

  .def("copy",[](const BSGlayer1b& x){return x.copy();})
  .def("copy",[](const BSGlayer1b& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",[](const BSGlayer1b& x){return x.zeros_like();})
  .def("randn_like",&BSGlayer1b::gaussian_like)

  .def_static("nrows",[](const BatchedGgraph& G, const Subgraph& S){
      return BSGlayer1b::nrows(G,S);})
  .def_static("n_gather_maps",[](const int k){
      return vector<int>({1,2,5})[k];})

  .def("get_grad",[](BSGlayer1b& x){return x.get_grad();})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](BSGlayer1b& r, const BSGlayer1b& x){r.add(x);})
  .def("cat_channels",[](const BSGlayer1b& x, const BSGlayer1b& y){
      cnine::fnlog timer("BatchedSubgraphLayer1b::cat_channels()");
      return cat_channels_sg(x,y);})
  .def("mprod",[](const BSGlayer1b& x, at::Tensor& M){
      return mprod_sg(x,ATview<float>(M));})
  .def("linear",[](const BSGlayer1b& x, at::Tensor& y, at::Tensor& b){
      cnine::fnlog timer("BatchedSubgraphLayer1b::linear()");
      return linear_sg(x,ATview<float>(y),ATview<float>(b));})
  .def("ReLU",[](const BSGlayer1b& x, const float alpha){
      cnine::fnlog timer("BatchedSubgraphLayer1b::ReLU()");
      return ReLU_sg(x,alpha);})


// ---- Linmaps -----------------------------------------------------------------------------------------------


//  .def_static("linmaps",[](const BSGlayer0b& x){
//      return BSGlayer1b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer1b& x){
      return BSGlayer1b::linmaps(x);}) 
//.def_static("linmaps",[](const BSGlayer2b& x){
//    return BSGlayer1b::linmaps(x);}) 

//  .def("add_linmaps_back",[](BSGlayer1b& x, BSGlayer0b& g){
//      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](BSGlayer1b& x, BSGlayer1b& g){
      x.add_linmaps_back_alt(g);})
//  .def("add_linmaps_back",[](BSGlayer1b& x, BSGlayer2b& g){
//      x.get_grad().add_linmaps_back(g.get_grad());})


// ---- Message passing --------------------------------------------------------------------------------------


  .def(pybind11::init([](const BSGlayer0b& x, const Subgraph& S){
	cnine::fnlog timer("BatchedSubgraphLayer1b::init(BatchedPtensors0b)");
	//cnine::tracer fn_tracer("BatchedSubgraphLayer1b from BSGlayer0b");
	return BatchedSubgraphLayer1b<float>(x,S);}))
  .def(pybind11::init([](const BSGlayer1b& x, const Subgraph& S){
	cnine::fnlog timer("BatchedSubgraphLayer1b::init(BatchedPtensors1b)");
	//cnine::tracer fn_tracer("BatchedSubgraphLayer1b from BSGlayer1b");
	return BatchedSubgraphLayer1b<float>(x,S);}))
  .def(pybind11::init([](const BSGlayer2b& x, const Subgraph& S){
	cnine::fnlog timer("BatchedSubgraphLayer1b::init(BatchedPtensors2b)");
	//cnine::tracer fn_tracer("BatchedSubgraphLayer1b from BSGlayer2b");
	return BatchedSubgraphLayer1b<float>(x,S);}))


  .def(pybind11::init([](const BatchedPtensors0b<float>& x, const BatchedGgraph& G, const Subgraph& S){
	cnine::fnlog timer("BatchedSubgraphLayer1b::init(BatchedPtensors0b)");
	//cnine::tracer fn_tracer("BatchedSubgraphLayer1b from BatchedPtensors0b");
	return BatchedSubgraphLayer1b<float>(x,G,S);
      }))
  .def(pybind11::init([](const BatchedPtensors1b<float>& x, const BatchedGgraph& G, const Subgraph& S){
	cnine::fnlog timer("BatchedSubgraphLayer1b::init(BatchedPtensors1b)");
	//cnine::tracer fn_tracer("BatchedSubgraphLayer1b from BatchedPtensors1b");
	return BatchedSubgraphLayer1b<float>(x,G,S);
      }))
  .def(pybind11::init([](const BatchedPtensors2b<float>& x, const BatchedGgraph& G, const Subgraph& S){
	cnine::fnlog timer("BatchedSubgraphLayer1b::init(BatchedPtensors2b)");
	//cnine::tracer fn_tracer("BatchedSubgraphLayer1b from BatchedPtensors2b");
	return BatchedSubgraphLayer1b<float>(x,G,S);
      }))


  .def("autobahn",[](const BSGlayer1b& x, at::Tensor& W, at::Tensor& B){
      cnine::fnlog timer("BatchedSubgraphLayer1b::autobahn()");
      return x.autobahn(ATview<float>(W),ATview<float>(B));})
  .def("add_autobahn_back0",[](BSGlayer1b& x, BSGlayer1b& r, at::Tensor& W){
      cnine::fnlog timer("BatchedSubgraphLayer1b::autobahn_back0()");
      x.add_autobahn_back0(r.get_grad(),ATview<float>(W));})
  .def("autobahn_back1",[](BSGlayer1b& x, at::Tensor& W, at::Tensor& B, BSGlayer1b& r){
      cnine::fnlog timer("BatchedSubgraphLayer1b::autobahn_back1()");
      x.add_autobahn_back1_to(ATview<float>(W), ATview<float>(B),r.get_grad());})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&BSGlayer1b::str,py::arg("indent")="")
  .def("__str__",&BSGlayer1b::str,py::arg("indent")="")
  .def("__repr__",&BSGlayer1b::repr);



//.def("get_dev",&BSGlayer1b::get_dev)
//.def("get_nc",&BSGlayer1b::get_nc)
//.def("get_atoms",[](const BSGlayer1b& x){return x.atoms.as_vecs();})
//.def("linmaps0",[](const BSGlayer1b& x){return sglinmaps0(x);})
//.def("linmaps1",[](const BSGlayer1b& x){return sglinmaps1(x);})
//.def("linmaps2",[](const BSGlayer1b& x){return sglinmaps2(x);})

//.def("gather",[](const BSGlayer1b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const BSGlayer1b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const BSGlayer2b& x, const Subgraph& a){return gather0(x,a);});
//.def(pybind11::init<ptens::Ggraph&, const at::Tensor&>())

//.def_static("create",[](const BatchedGgraph& G, const int _nc, const int fcode, const int _dev){
//    return BSGlayer1b(G,_nc,fcode,_dev);}, 
//  py::arg("graph"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

//.def("to",[](const BSGlayer1b& x, const int dev){return BSGlayer1b(x,dev);})
//.def("to_device",[](BSGlayer1b& x, const int dev){return BSGlayer1b(x,dev);})

//.def("cat",&BSGlayer1b::cat)
//.def("scale_channels",[](BSGlayer1b& x, at::Tensor& y){
//      return scale_channels_sg(x,ATview<float>(y));})
//  .def(pybind11::init([](at::Tensor& R, const BatchedPtensors0b<float>& x, const BatchedGgraph& G, const Subgraph& S){
//	cnine::fnlog timer("BatchedSubgraphLayer1b(ATen)::init(BatchedPtensors0b)");
//	cnine::tracer fn_tracer("BatchedSubgraphLayer1b(ATen) from BatchedPtensors0b");
//	return BatchedSubgraphLayer1b<float>(R.data<float>(),x,G,S);
//      }))
//  .def(pybind11::init([](at::Tensor& R, const BatchedPtensors1b<float>& x, const BatchedGgraph& G, const Subgraph& S){
//	cnine::fnlog timer("BatchedSubgraphLayer1b(ATen)::init(BatchedPtensors1b)");
//	cnine::tracer fn_tracer("BatchedSubgraphLayer1b(ATen) from BatchedPtensors1b");
//	return BatchedSubgraphLayer1b<float>(R.data<float>(),x,G,S);
//      }))

