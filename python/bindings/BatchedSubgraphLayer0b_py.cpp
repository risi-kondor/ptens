typedef BatchedSubgraphLayer0b<float> BSGlayer0b;
typedef BatchedSubgraphLayer1b<float> BSGlayer1b;
typedef BatchedSubgraphLayer2b<float> BSGlayer2b;


pybind11::class_<BSGlayer0b,BatchedPtensors0b<float> >(m,"batched_subgraphlayer0b")

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


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](BSGlayer0b& r, const BSGlayer0b& x){r.add(x);})

  .def("cat_channels",[](const BSGlayer0b& x, const BSGlayer0b& y){
      cnine::fnlog timer("BatchedSubgraphLayer0b::cat_channels()");
      return cat_channels_sg(x,y);})
  .def("mprod",[](const BSGlayer0b& x, at::Tensor& M){
      return mprod_sg(x,ATview<float>(M));})
  .def("linear",[](const BSGlayer0b& x, at::Tensor& y, at::Tensor& b){
      cnine::fnlog timer("BatchedSubgraphLayer0b::linear()");
      return linear_sg(x,ATview<float>(y),ATview<float>(b));})
  .def("ReLU",[](const BSGlayer0b& x, const float alpha){
      cnine::fnlog timer("BatchedSubgraphLayer0b::ReLU()");
      return ReLU_sg(x,alpha);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const BSGlayer0b& x){
      return BSGlayer0b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer1b& x){
      return BSGlayer0b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer2b& x){
      return BSGlayer0b::linmaps(x);}) 

  .def(pybind11::init([](const BSGlayer0b& x, const Subgraph& S, const int min_overlaps){
	cnine::fnlog timer("BatchedSubgraphLayer0b::init(BatchedPtensors0b)");
	return BatchedSubgraphLayer0b<float>(x,S,min_overlaps);}))
  .def(pybind11::init([](const BSGlayer1b& x, const Subgraph& S, const int min_overlaps){
	cnine::fnlog timer("BatchedSubgraphLayer0b::init(BatchedPtensors1b)");
	return BatchedSubgraphLayer0b<float>(x,S,min_overlaps);}))
  .def(pybind11::init([](const BSGlayer2b& x, const Subgraph& S, const int min_overlaps){
	cnine::fnlog timer("BatchedSubgraphLayer0b::init(BatchedPtensors2b)");
	return BatchedSubgraphLayer0b<float>(x,S);}))

  .def(pybind11::init([](const BatchedPtensors0b<float>& x, const BatchedGgraph& G, const Subgraph& S, const int min_overlaps){
	cnine::fnlog timer("BatchedSubgraphLayer0b::init(BatchedPtensors0b)");
	return BatchedSubgraphLayer0b<float>(x,G,S,min_overlaps);
      }))
  .def(pybind11::init([](const BatchedPtensors1b<float>& x, const BatchedGgraph& G, const Subgraph& S, const int min_overlaps){
	cnine::fnlog timer("BatchedSubgraphLayer0b::init(BatchedPtensors1b)");
	return BatchedSubgraphLayer0b<float>(x,G,S,min_overlaps);
      }))
  .def(pybind11::init([](const BatchedPtensors2b<float>& x, const BatchedGgraph& G, const Subgraph& S, const int min_overlaps){
	cnine::fnlog timer("BatchedSubgraphLayer0b::init(BatchedPtensors2b)");
	return BatchedSubgraphLayer0b<float>(x,G,S,min_overlaps);
      }))


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&BSGlayer0b::str,py::arg("indent")="")
  .def("__str__",&BSGlayer0b::str,py::arg("indent")="")
  .def("__repr__",&BSGlayer0b::repr);


