typedef BatchedPtensors2b<float> BPtensors2; 
typedef cnine::ATview<float> TVIEW;

pybind11::class_<BatchedPtensors2b<float> >(m,"batched_ptensors2b")

  .def(py::init([](const vector<vector<vector<int> > >& atoms, at::Tensor& M){
	return BPtensors2(BatchedAtomsPack(atoms),Tensor<float>(M));}))

  .def_static("create",[](const vector<vector<vector<int> > >_atoms, const int _nc, const int fcode, const int _dev){
      return BPtensors2(BatchedAtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const BatchedAtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return BPtensors2(_atoms,_nc,fcode,_dev);})
  
  .def("like",[](const BPtensors2& x, at::Tensor& M){
      return BPtensors2(x.atoms,ATview<float>(M));})
  .def("copy",[](const BPtensors2& x){return x.copy();})
  .def("copy",[](const BPtensors2& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",[](const BPtensors2& x){return BPtensors2::zeros_like(x);})
  .def("randn_like",[](const BPtensors2& x){return BPtensors2::gaussian_like(x);})


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("__len__",&BPtensors2::size)
  .def("add_to_grad",[](BPtensors2& x, at::Tensor& y){x.get_grad().add(ATview<float>(y));})
  .def("add_to_grad",[](BPtensors2& x, const BPtensors2& y, const float c){x.get_grad().add(y,c);})
  .def("get_grad",[](BPtensors2& x){return x.get_grad();})

  .def("__getitem__",[](const BPtensors2& x, const int i){return x[i];})
  .def("torch",[](const BPtensors2& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&BPtensors2::get_dev)
  .def("get_nc",&BPtensors2::get_nc)
  .def("get_atoms",[](const BPtensors2& x){return x.get_atoms();})
  .def("dim",&BPtensors2::dim)

//.def("to",[](const BPtensors2& x, const int dev){return BPtensors2(x,dev);})
//.def("to_device",&BPtensors2::move_to_device)


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](BPtensors2& r, const BPtensors2& x){r.add(x);})
  .def("add_back",[](BPtensors2& x, const BPtensors2& g){x.add_to_grad(g.get_grad());})

  .def("cat_channels",[](const BPtensors2& x, const BPtensors2& y){return cat_channels(x,y);})
  .def("cat_channels_back0",[](BPtensors2& x, const BPtensors2& r){return x.cat_channels_back0(r);})
  .def("cat_channels_back1",[](BPtensors2& x, const BPtensors2& r){return x.cat_channels_back1(r);})

//  .def("cat",&BPtensors2::cat)
//  .def("add_cat_back",[](BPtensors2& x, BPtensors2& r, const int offs){
//      x.get_grad()+=r.get_grad().rows(offs,x.dim(0));})

//.def("outer",&BatchedPtensors2b::outer)
//.def("outer_back0",&BatchedPtensors2b::outer_back0)
//.def("outer_back1",&BatchedPtensors2b::outer_back1)

  .def("scale_channels",[](BPtensors2& x, at::Tensor& y){
      return scale_channels(x,ATview<float>(y));})
  .def("add_scale_channels_back0",[](BPtensors2& r, const BPtensors2& g, at::Tensor& y){
      r.add_scale_channels_back(g,ATview<float>(y));})

  .def("mprod",[](const BPtensors2& x, at::Tensor& M){
      return mprod(x,ATview<float>(M));})
  .def("add_mprod_back0",[](BPtensors2& r, const BPtensors2& g, at::Tensor& M){
      r.add_mprod_back0(g,ATview<float>(M));})
  .def("mprod_back1",[](const BPtensors2& x, const BPtensors2& g){
      return (x.transp()*g.get_grad()).torch();})

  .def("linear",[](const BPtensors2& x, at::Tensor& y, at::Tensor& b){
      return linear(x,ATview<float>(y),ATview<float>(b));})
  .def("add_linear_back0",[](BPtensors2& r, const BPtensors2& g, at::Tensor& y){
      r.add_linear_back0(g,ATview<float>(y));})
  .def("linear_back1",[](const BPtensors2& x, const BPtensors2& g){
      return (x.transp()*g.get_grad()).torch();})
  .def("linear_back2",[](const BPtensors2& x, BPtensors2& g){
      return g.get_grad().sum(0).torch();})

  .def("ReLU",[](const BPtensors2& x, const float alpha){
      return ReLU(x,alpha);})
  .def("add_ReLU_back",[](BPtensors2& x, const BPtensors2& g, const float alpha){
      x.add_ReLU_back(g,alpha);})

  .def("inp",[](const BPtensors2& x, const BPtensors2& y){return x.inp(y);})
  .def("diff2",[](const BPtensors2& x, const BPtensors2& y){return x.diff2(y);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const BatchedPtensors0b<float>& x){
      return BPtensors2::linmaps(x);}) 
  .def_static("linmaps",[](const BatchedPtensors1b<float>& x){
      return BPtensors2::linmaps(x);}) 
  .def_static("linmaps",[](const BatchedPtensors2b<float>& x){
      return BPtensors2::linmaps(x);}) 

  .def_static("gather",[](const BatchedPtensors0b<float>& x, const BatchedAtomsPack& a){
      return BPtensors2::gather(x,a);}) 
  .def_static("gather",[](const BatchedPtensors1b<float>& x, const BatchedAtomsPack& a){
      return BPtensors2::gather(x,a);}) 
  .def_static("gather",[](const BatchedPtensors2b<float>& x, const BatchedAtomsPack& a){
      return BPtensors2::gather(x,a);}) 

  .def("add_linmaps_back",[](BPtensors2& x, BatchedPtensors0b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](BPtensors2& x, BatchedPtensors1b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](BPtensors2& x, BatchedPtensors2b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})

  .def("add_gather_back",[](BPtensors2& x, BatchedPtensors0b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](BPtensors2& x, BatchedPtensors1b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back_alt",[](BPtensors2& x, BatchedPtensors1b<float>& g){
      cnine::fnlog timer("BatchedPtensors2b::gather1_back()");
      x.add_gather_back_alt(g);})
  .def("add_gather_back",[](BPtensors2& x, BatchedPtensors2b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&BPtensors2::str,py::arg("indent")="")
  .def("__str__",&BPtensors2::str,py::arg("indent")="")
  .def("__repr__",&BPtensors2::repr);


