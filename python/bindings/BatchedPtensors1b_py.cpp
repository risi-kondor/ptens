typedef BatchedPtensors1b<float> BPtensors1; 
typedef cnine::ATview<float> TVIEW;

pybind11::class_<BatchedPtensors1b<float> >(m,"batched_ptensors1b")

//.def(py::init([](at::Tensor& M){
//	return BPtensors1(Ltensor<float>(TVIEW(M)));}))

//  .def(py::init([](const BatchedAtomsPack& atoms, vector<at::Tensor>& M){
//	vector<TVIEW> v;
//	for(int i=0; i<M.size(); i++) v.push_back(TVIEW(M[i]));
//	return BPtensors1(atoms,v);}))

  .def(py::init([](const vector<vector<vector<int> > >& atoms, at::Tensor& M){
	return BPtensors1(BatchedAtomsPack(atoms),Ltensor<float>(M));})) // maybe should remove ATview from other too?

//  .def_static("create",[](const int n, const int _nc, const int fcode, const int _dev){
//      return BPtensors1(n,_nc,fcode,_dev);}, 
//    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const vector<vector<vector<int> > > _atoms, const int _nc, const int fcode, const int _dev){
      return BPtensors1(BatchedAtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const BatchedAtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return BPtensors1(_atoms,_nc,fcode,_dev);})
  
  .def("like",[](const BPtensors1& x, at::Tensor& M){
      return BPtensors1(x.atoms,ATview<float>(M));})
  .def("copy",[](const BPtensors1& x){return x.copy();})
  .def("copy",[](const BPtensors1& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",[](const BPtensors1& x){return BPtensors1::zeros_like(x);})
  .def("randn_like",[](const BPtensors1& x){return BPtensors1::gaussian_like(x);})


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("__len__",&BPtensors1::size)
//.def("add_to_grad",[](BPtensors1& x, at::Tensor& y){x.add_to_grad(ATview<float>(y));})
//.def("add_to_grad",[](BPtensors1& x, const BPtensors1& y, const float c){x.add_to_grad(y,c);})
  .def("add_to_grad",[](BPtensors1& x, at::Tensor& y){x.get_grad().add(ATview<float>(y));})
  .def("add_to_grad",[](BPtensors1& x, const BPtensors1& y, const float c){x.get_grad().add(y,c);})
  .def("get_grad",[](BPtensors1& x){return x.get_grad();})

  .def("__getitem__",[](const BPtensors1& x, const int i){return x[i];})
  .def("torch",[](const BPtensors1& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&BPtensors1::get_dev)
  .def("get_nc",&BPtensors1::get_nc)
//.def("get_atoms",[](const BPtensors1& x){return x.get_atoms();})
  .def("dim",&BPtensors1::dim)

//.def("to",[](const BPtensors1& x, const int dev){return BPtensors1(x,dev);})
//.def("to_device",&BPtensors1::move_to_device)


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](BPtensors1& r, const BPtensors1& x){r.add(x);})
  .def("add_back",[](BPtensors1& x, const BPtensors1& g){x.add_to_grad(g.get_grad());})

//  .def("cat_channels",[](const BPtensors1& x, const BPtensors1& y){return cat_channels(x,y);})
//  .def("cat_channels_back0",&BPtensors1::cat_channels_back0)
//  .def("cat_channels_back1",&BPtensors1::cat_channels_back1)

  .def("cat",&BPtensors1::cat)
  .def("add_cat_back",[](BPtensors1& x, BPtensors1& r, const int offs){
      x.get_grad()+=r.get_grad().rows(offs,x.dim(0));})

//.def("outer",&BatchedPtensors1b::outer)
//.def("outer_back0",&BatchedPtensors1b::outer_back0)
//.def("outer_back1",&BatchedPtensors1b::outer_back1)

  .def("scale_channels",[](BPtensors1& x, at::Tensor& y){
      return scale_channels(x,ATview<float>(y));})
  .def("add_scale_channels_back0",[](BPtensors1& r, const BPtensors1& g, at::Tensor& y){
      r.add_scale_channels_back(g,ATview<float>(y));})

  .def("mprod",[](const BPtensors1& x, at::Tensor& M){
      return mprod(x,ATview<float>(M));})
  .def("add_mprod_back0",[](BPtensors1& r, const BPtensors1& g, at::Tensor& M){
      r.add_mprod_back0(g,ATview<float>(M));})
  .def("mprod_back1",[](const BPtensors1& x, const BPtensors1& g){
      return (x.transp()*g.get_grad()).torch();})

  .def("linear",[](const BPtensors1& x, at::Tensor& y, at::Tensor& b){
      return linear(x,ATview<float>(y),ATview<float>(b));})
  .def("add_linear_back0",[](BPtensors1& r, const BPtensors1& g, at::Tensor& y){
      r.add_linear_back0(g,ATview<float>(y));})
  .def("linear_back1",[](const BPtensors1& x, const BPtensors1& g){
      return (x.transp()*g.get_grad()).torch();})
  .def("linear_back2",[](const BPtensors1& x, BPtensors1& g){
      return g.get_grad().sum(0).torch();})

  .def("ReLU",[](const BPtensors1& x, const float alpha){
      return ReLU(x,alpha);})
  .def("add_ReLU_back",[](BPtensors1& x, const BPtensors1& g, const float alpha){
      x.add_ReLU_back(g,alpha);})

  .def("inp",[](const BPtensors1& x, const BPtensors1& y){return x.inp(y);})
  .def("diff2",[](const BPtensors1& x, const BPtensors1& y){return x.diff2(y);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const BatchedPtensors0b<float>& x){return BPtensors1::linmaps(x);}) 
  .def_static("linmaps",[](const BatchedPtensors1b<float>& x){return BPtensors1::linmaps(x);}) 
  .def_static("linmaps",[](const BatchedPtensors2b<float>& x){return BPtensors1::linmaps(x);}) 

  .def_static("gather",[](const BatchedPtensors0b<float>& x, const BatchedAtomsPack& a){
      return BPtensors1::gather(x,a);}) 
  .def_static("gather",[](const BatchedPtensors1b<float>& x, const BatchedAtomsPack& a){
      return BPtensors1::gather(x,a);}) 
  .def_static("gather",[](const BatchedPtensors2b<float>& x, const BatchedAtomsPack& a){
      return BPtensors1::gather(x,a);}) 

  .def("add_linmaps_back",[](BPtensors1& x, BatchedPtensors0b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](BPtensors1& x, BatchedPtensors1b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](BPtensors1& x, BatchedPtensors2b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})

  .def("add_gather_back",[](BPtensors1& x, BatchedPtensors0b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](BPtensors1& x, BatchedPtensors1b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](BPtensors1& x, BatchedPtensors2b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&BPtensors1::str,py::arg("indent")="")
  .def("__str__",&BPtensors1::str,py::arg("indent")="")
  .def("__repr__",&BPtensors1::repr);


