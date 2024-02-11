typedef BatchedSubgraphLayer0b<float> BSGlayer0b;
typedef BatchedSubgraphLayer1b<float> BSGlayer1b;
typedef BatchedSubgraphLayer2b<float> BSGlayer2b;


pybind11::class_<BSGlayer0b,BatchedPtensors0b<float> >(m,"batched_subgraphlayer0b")

//.def(pybind11::init<ptens::Ggraph&, const at::Tensor&>())

//.def_static("create",[](const BatchedGgraph& G, const int _nc, const int fcode, const int _dev){
//    return BSGlayer0b(G,_nc,fcode,_dev);}, 
//  py::arg("graph"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("from_vertex_features",[](const vector<int>& keys, at::Tensor&M){
      return BSGlayer0b::from_vertex_features(keys,ATview<float>(M));})

  .def_static("from_edge_features",[](const vector<int>& keys, at::Tensor&M){
      return BSGlayer0b::from_edge_features(keys,ATview<float>(M));})

  .def_static("like",[](const BSGlayer0b& x, at::Tensor& M){
      return BSGlayer0b(x.G,x.S,x.atoms,ATview<float>(M));})
  .def("copy",[](const BSGlayer0b& x){return x.copy();})
  .def("copy",[](const BSGlayer0b& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",&BSGlayer0b::zeros_like)
  .def("randn_like",&BSGlayer0b::gaussian_like)

//.def("to",[](const BSGlayer0b& x, const int dev){return BSGlayer0b(x,dev);})
//.def("to_device",[](BSGlayer0b& x, const int dev){return BSGlayer0b(x,dev);})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](BSGlayer0b& r, const BSGlayer0b& x){r.add(x);})

  .def("cat_channels",[](const BSGlayer0b& x, const BSGlayer0b& y){
      cnine::fnlog timer("BatchedSubgraphLayer0b::cat_channels()");
      return cat_channels_sg(x,y);})
//.def("cat",&BSGlayer0b::cat)
//.def("scale_channels",[](BSGlayer0b& x, at::Tensor& y){
//      return scale_channels_sg(x,ATview<float>(y));})
  .def("mprod",[](const BSGlayer0b& x, at::Tensor& M){
      return mprod_sg(x,ATview<float>(M));})
  .def("linear",[](const BSGlayer0b& x, at::Tensor& y, at::Tensor& b){
      cnine::fnlog timer("BatchedSubgraphLayer0b::linear()");
      return linear_sg(x,ATview<float>(y),ATview<float>(b));})
  .def("ReLU",[](const BSGlayer0b& x, const float alpha){
      cnine::fnlog timer("BatchedSubgraphLayer0b::ReLU()");
      return ReLU_sg(x,alpha);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const BSGlayer0b& x){return BSGlayer1b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer1b& x){return BSGlayer1b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer2b& x){return BSGlayer1b::linmaps(x);}) 

  .def(pybind11::init([](const BSGlayer0b& x, const Subgraph& S){
	cnine::fnlog timer("BatchedSubgraphLayer0b::init(BatchedPtensors0b)");
	cnine::tracer fn_tracer("BatchedSubgraphLayer0b from BSGlayer0b");
	return BatchedSubgraphLayer0b<float>(x,S);}))
  .def(pybind11::init([](const BSGlayer1b& x, const Subgraph& S){
	cnine::fnlog timer("BatchedSubgraphLayer0b::init(BatchedPtensors1b)");
	cnine::tracer fn_tracer("BatchedSubgraphLayer0b from BSGlayer1b");
	return BatchedSubgraphLayer0b<float>(x,S);}))
  .def(pybind11::init([](const BSGlayer2b& x, const Subgraph& S){
	cnine::fnlog timer("BatchedSubgraphLayer0b::init(BatchedPtensors2b)");
	cnine::tracer fn_tracer("BatchedSubgraphLayer0b from BSGlayer2b");
	return BatchedSubgraphLayer0b<float>(x,S);}))

//.def(pybind11::init<const BSGlayer0b&, const Subgraph&>())
//.def(pybind11::init<const BSGlayer1b&, const Subgraph&>())
//.def(pybind11::init<const BSGlayer2b&, const Subgraph&>())

  .def(pybind11::init<const BatchedPtensors0b<float>&, const BatchedGgraph&, const Subgraph&>())
  .def(pybind11::init<const BatchedPtensors1b<float>&, const BatchedGgraph&, const Subgraph&>())
  .def(pybind11::init<const BatchedPtensors2b<float>&, const BatchedGgraph&, const Subgraph&>());


// ---- I/O --------------------------------------------------------------------------------------------------

//.def("str",&BSGlayer0b::str,py::arg("indent")="")
//.def("__str__",&BSGlayer0b::str,py::arg("indent")="")
//.def("__repr__",&BSGlayer0b::repr);



//.def("get_dev",&BSGlayer0b::get_dev)
//.def("get_nc",&BSGlayer0b::get_nc)
//.def("get_atoms",[](const BSGlayer0b& x){return x.atoms.as_vecs();})
//.def("linmaps0",[](const BSGlayer0b& x){return sglinmaps0(x);})
//.def("linmaps1",[](const BSGlayer0b& x){return sglinmaps1(x);})
//.def("linmaps2",[](const BSGlayer0b& x){return sglinmaps2(x);})

//.def("gather",[](const BSGlayer0b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const BSGlayer1b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const BSGlayer2b& x, const Subgraph& a){return gather0(x,a);});
