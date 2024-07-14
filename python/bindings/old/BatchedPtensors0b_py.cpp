typedef BatchedPtensors0b<float> BPtensors0; 
typedef cnine::ATview<float> TVIEW;

pybind11::class_<BatchedPtensors0b<float> >(m,"batched_ptensors0b")

  .def(py::init([](at::Tensor& M, const vector<int>& sizes){
	return BPtensors0(Tensor<float>(M),sizes);}))
  
  .def_static("from_tensors",[](at::Tensor& M, const vector<int>& sizes){
      return BPtensors0(Tensor<float>(M),sizes);})

  .def(py::init([](const vector<vector<vector<int> > >& atoms, at::Tensor& M){
	return BPtensors0(BatchedAtomsPack(atoms),Tensor<float>(M));}))

  .def_static("create",[](const vector<vector<vector<int> > >_atoms, const int _nc, const int fcode, const int _dev){
      return BPtensors0(BatchedAtomsPack(_atoms),_nc,fcode,_dev);}) 

  .def_static("create",[](const BatchedAtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return BPtensors0(_atoms,_nc,fcode,_dev);})
  
  .def("like",[](const BPtensors0& x, at::Tensor& M){
      return BPtensors0(x.atoms,ATview<float>(M));})
  .def("copy",[](const BPtensors0& x){return x.copy();})
  .def("copy",[](const BPtensors0& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",[](const BPtensors0& x){return BPtensors0::zeros_like(x);})
  .def("randn_like",[](const BPtensors0& x){return BPtensors0::gaussian_like(x);})


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("getk",[](const BPtensors0& x){return 0;})
  .def("__len__",&BPtensors0::size)
  .def("add_to_grad",[](BPtensors0& x, at::Tensor& y){x.get_grad().add(ATview<float>(y));})
  .def("add_to_grad",[](BPtensors0& x, const BPtensors0& y, const float c){x.get_grad().add(y,c);})
  .def("get_grad",[](BPtensors0& x){return x.get_grad();})

  .def("__getitem__",[](const BPtensors0& x, const int i){return x[i];})
  .def("torch",[](const BPtensors0& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&BPtensors0::get_dev)
  .def("get_nc",&BPtensors0::get_nc)
  .def("get_atoms",[](const BPtensors0& x){return x.get_atoms();})
  .def("dim",&BPtensors0::dim)

//.def("to",[](const BPtensors0& x, const int dev){return BPtensors0(x,dev);})
//.def("to_device",&BPtensors0::move_to_device)


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](BPtensors0& r, const BPtensors0& x){r.add(x);})
  .def("add_back",[](BPtensors0& x, const BPtensors0& g){x.add_to_grad(g.get_grad());})

  .def("cat_channels",[](const BPtensors0& x, const BPtensors0& y){
      cnine::fnlog timer("BatchedSubgraphLayer0b::cat_channels()");
      return cat_channels(x,y);})
  .def("cat_channels_back0",[](BPtensors0& x, const BPtensors0& r){
      cnine::fnlog timer("BatchedSubgraphLayerb::cat_channels_back0()");
      return x.cat_channels_back0(r);})
  .def("cat_channels_back1",[](BPtensors0& x, const BPtensors0& r){
      cnine::fnlog timer("BatchedSubgraphLayerb::cat_channels_back1()");
      return x.cat_channels_back1(r);})

  .def("scale_channels",[](BPtensors0& x, at::Tensor& y){
      return scale_channels(x,ATview<float>(y));})
  .def("add_scale_channels_back0",[](BPtensors0& r, const BPtensors0& g, at::Tensor& y){
      r.add_scale_channels_back(g,ATview<float>(y));})

  .def("mprod",[](const BPtensors0& x, at::Tensor& M){
      cnine::fnlog timer("BatchedPtensors1b::mprod()");
      return mprod(x,ATview<float>(M));})
  .def("add_mprod_back0",[](BPtensors0& r, const BPtensors0& g, at::Tensor& M){
      cnine::fnlog timer("BatchedPtensors1b::mprod_back0()");
      r.add_mprod_back0(g,ATview<float>(M));})
  .def("mprod_back1",[](const BPtensors0& x, const BPtensors0& g){
      cnine::fnlog timer("BatchedPtensors1b::mprod_back1()");
      return (x.transp()*g.get_grad()).torch();})

  .def("linear",[](const BPtensors0& x, at::Tensor& y, at::Tensor& b){
      cnine::fnlog timer("BatchedPtensors0b::linear()");
      return linear(x,ATview<float>(y),ATview<float>(b));})
  .def("add_linear_back0",[](BPtensors0& r, const BPtensors0& g, at::Tensor& y){
      cnine::fnlog timer("BatchedPtensors0b::linear_back0()");
      r.add_linear_back0(g,ATview<float>(y));})
  .def("linear_back1",[](const BPtensors0& x, const BPtensors0& g){
      cnine::fnlog timer("BatchedPtensors1b::linear_back0()");
      return (x.transp()*g.get_grad()).torch();})
  .def("linear_back2",[](const BPtensors0& x, BPtensors0& g){
      cnine::fnlog timer("BatchedPtensors0b::linear_back2()");
      auto& p=g.get_grad();
      Tensor<float> xg({p.dim(1)},0,p.get_dev());
      p.view2().reduce0_destructively_into(xg.view1());
      return xg.torch();
    })

  .def("ReLU",[](const BPtensors0& x, const float alpha){
      cnine::fnlog timer("BatchedPtensors0b::add_ReLU()");
      return ReLU(x,alpha);})
  .def("add_ReLU_back",[](BPtensors0& x, const BPtensors0& g, const float alpha){
      cnine::fnlog timer("BatchedPtensors0b::add_ReLU_back()");
      x.add_ReLU_back(g,alpha);})

  .def("inp",[](const BPtensors0& x, const BPtensors0& y){return x.inp(y);})
  .def("diff2",[](const BPtensors0& x, const BPtensors0& y){return x.diff2(y);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const BPtensors0& x){
      return BPtensors0::linmaps(x);}) 
  .def_static("linmaps",[](const BatchedPtensors1b<float>& x){
      return BPtensors0::linmaps(x);}) 
  .def_static("linmaps",[](const BatchedPtensors2b<float>& x){
      return BPtensors0::linmaps(x);}) 

  .def_static("gather",[](const BPtensors0& x, const BatchedAtomsPack& a, const int min_overlaps){
      return BPtensors0::gather(x,a,min_overlaps);}) 
  .def_static("gather",[](const BatchedPtensors1b<float>& x, const BatchedAtomsPack& a, const int min_overlaps){
      return BPtensors0::gather(x,a,min_overlaps);}) 
  .def_static("gather",[](const BatchedPtensors2b<float>& x, const BatchedAtomsPack& a, const int min_overlaps){
      return BPtensors0::gather(x,a);}) 

  .def("add_linmaps_back",[](BPtensors0& x, BPtensors0& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](BPtensors0& x, BatchedPtensors1b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](BPtensors0& x, BatchedPtensors2b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})

  .def("add_gather_back",[](BPtensors0& x, BPtensors0& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back_alt",[](BPtensors0& x, BPtensors0& g){
      x.add_gather_back_alt(g);})
  .def("add_gather_back",[](BPtensors0& x, BatchedPtensors1b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back_alt",[](BPtensors0& x, BatchedPtensors1b<float>& g){
      x.add_gather_back_alt(g);})
  .def("add_gather_back",[](BPtensors0& x, BatchedPtensors2b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&BPtensors0::str,py::arg("indent")="")
  .def("__str__",&BPtensors0::str,py::arg("indent")="")
  .def("__repr__",&BPtensors0::repr);


