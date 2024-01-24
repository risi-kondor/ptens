typedef cnine::ATview<float> TVIEW;

pybind11::class_<BatchedBatchedPtensors0b<float> >(m,"batched_ptensors0b")

  .def(py::init([](at::Tensor& M){
	return BatchedPtensors0b(Ltensor<float>(TVIEW(M)));}))
  .def(py::init([](const AtomsPack& atoms, at::Tensor& M){
	return BatchedPtensors0b(atoms,Ltensor<float>(TVIEW(M)));}))
  .def(py::init([](const vector<vector<int> >& atoms, at::Tensor& M){
	return BatchedPtensors0b(AtomsPack(atoms),Ltensor<float>(TVIEW(M)));}))

  .def_static("create",[](const int n, const int _nc, const int fcode, const int _dev){
      return BatchedPtensors0b<float>(n,_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const vector<vector<int> > _atoms, const int _nc, const int fcode, const int _dev){
      return BatchedPtensors0b<float>(AtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const AtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return BatchedPtensors0b<float>(_atoms,cnine::channels=_nc,cnine::filltype=fcode,cnine::device=_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def("like",[](const BatchedPtensors0b<float>& x, at::Tensor& M){
      return BatchedPtensors0b(x.atoms,ATview<float>(M));})
  .def("copy",[](const BatchedPtensors0b<float>& x){return x.copy();})
  .def("copy",[](const BatchedPtensors0b<float>& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",[](const BatchedPtensors0b<float>& x){return BatchedPtensors0b<float>::zeros_like(x);})
  .def("randn_like",[](const BatchedPtensors0b<float>& x){return BatchedPtensors0b<float>::gaussian_like(x);})


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("__len__",&BatchedPtensors0b<float>::size)
//.def("add_to_grad",[](BatchedPtensors0b<float>& x, at::Tensor& y){x.add_to_grad(ATview<float>(y));})
//.def("add_to_grad",[](BatchedPtensors0b<float>& x, const BatchedPtensors0b<float>& y, const float c){x.add_to_grad(y,c);})
  .def("add_to_grad",[](BatchedPtensors0b<float>& x, at::Tensor& y){x.get_grad().add(ATview<float>(y));})
  .def("add_to_grad",[](BatchedPtensors0b<float>& x, const BatchedPtensors0b<float>& y, const float c){x.get_grad().add(y,c);})
  .def("get_grad",[](BatchedPtensors0b<float>& x){return x.get_grad();})

  .def("__getitem__",[](const BatchedPtensors0b<float>& x, const int i){return x(i);})
  .def("torch",[](const BatchedPtensors0b<float>& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&BatchedPtensors0b<float>::get_dev)
  .def("get_nc",&BatchedPtensors0b<float>::get_nc)
  .def("get_atoms",[](const BatchedPtensors0b<float>& x){return x.get_atoms();})
  .def("dim",&BatchedPtensors0b<float>::dim)

  .def("to",[](const BatchedPtensors0b<float>& x, const int dev){return BatchedPtensors0b<float>(x,dev);})
  .def("to_device",&BatchedPtensors0b<float>::move_to_device)


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](BatchedPtensors0b<float>& r, const BatchedPtensors0b<float>& x){r.add(x);})
  .def("add_back",[](BatchedPtensors0b<float>& x, const BatchedPtensors0b<float>& g){x.add_to_grad(g.get_grad());})

  .def("cat_channels",[](const BatchedPtensors0b<float>& x, const BatchedPtensors0b<float>& y){return cat_channels(x,y);})
  .def("cat_channels_back0",&BatchedPtensors0b<float>::cat_channels_back0)
  .def("cat_channels_back1",&BatchedPtensors0b<float>::cat_channels_back1)

  .def("cat",&BatchedPtensors0b<float>::cat)
  .def("add_cat_back",[](BatchedPtensors0b<float>& x, BatchedPtensors0b<float>& r, const int offs){
      x.get_grad()+=r.get_grad().rows(offs,x.dim(0));})

//.def("outer",&BatchedPtensors0b::outer)
//.def("outer_back0",&BatchedPtensors0b::outer_back0)
//.def("outer_back1",&BatchedPtensors0b::outer_back1)

  .def("scale_channels",[](BatchedPtensors0b<float>& x, at::Tensor& y){
      return scale_channels(x,ATview<float>(y));})
  .def("add_scale_channels_back0",[](BatchedPtensors0b<float>& r, const BatchedPtensors0b<float>& g, at::Tensor& y){
      r.add_scale_channels_back(g,ATview<float>(y));})

  .def("mprod",[](const BatchedPtensors0b<float>& x, at::Tensor& M){
      return mprod(x,ATview<float>(M));})
  .def("add_mprod_back0",[](BatchedPtensors0b<float>& r, const BatchedPtensors0b<float>& g, at::Tensor& M){
      r.add_mprod_back0(g,ATview<float>(M));})
  .def("mprod_back1",[](const BatchedPtensors0b<float>& x, const BatchedPtensors0b<float>& g){
      return (x.transp()*g.get_grad()).torch();})

  .def("linear",[](const BatchedPtensors0b<float>& x, at::Tensor& y, at::Tensor& b){
      return linear(x,ATview<float>(y),ATview<float>(b));})
  .def("add_linear_back0",[](BatchedPtensors0b<float>& r, const BatchedPtensors0b<float>& g, at::Tensor& y){
      r.add_linear_back0(g,ATview<float>(y));})
  .def("linear_back1",[](const BatchedPtensors0b<float>& x, const BatchedPtensors0b<float>& g){
      return (x.transp()*g.get_grad()).torch();})
  .def("linear_back2",[](const BatchedPtensors0b<float>& x, BatchedPtensors0b<float>& g){
      return g.get_grad().sum(0).torch();})

  .def("ReLU",[](const BatchedPtensors0b<float>& x, const float alpha){
      return ReLU(x,alpha);})
  .def("add_ReLU_back",[](BatchedPtensors0b<float>& x, const BatchedPtensors0b<float>& g, const float alpha){
      x.add_ReLU_back(g,alpha);})

  .def("inp",[](const BatchedPtensors0b<float>& x, const BatchedPtensors0b<float>& y){return x.inp(y);})
  .def("diff2",[](const BatchedPtensors0b<float>& x, const BatchedPtensors0b<float>& y){return x.diff2(y);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const BatchedPtensors0b<float>& x){
      return BatchedPtensors0b<float>::linmaps(x);}) 
  .def_static("linmaps",[](const Ptensors1b<float>& x){
      return BatchedPtensors0b<float>::linmaps(x);}) 
  .def_static("linmaps",[](const Ptensors2b<float>& x){
      return BatchedPtensors0b<float>::linmaps(x);}) 

  .def_static("gather",[](const BatchedPtensors0b<float>& x, const AtomsPack& a){
      return BatchedPtensors0b<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors1b<float>& x, const AtomsPack& a){
      return BatchedPtensors0b<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors2b<float>& x, const AtomsPack& a){
      return Ptensors1b<float>::gather(x,a);}) 

  .def("add_linmaps_back",[](BatchedPtensors0b<float>& x, BatchedPtensors0b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](BatchedPtensors0b<float>& x, Ptensors1b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](BatchedPtensors0b<float>& x, Ptensors2b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})

  .def("add_gather_back",[](BatchedPtensors0b<float>& x, BatchedPtensors0b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](BatchedPtensors0b<float>& x, Ptensors1b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](BatchedPtensors0b<float>& x, Ptensors2b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&BatchedPtensors0b<float>::str,py::arg("indent")="")
  .def("__str__",&BatchedPtensors0b<float>::str,py::arg("indent")="")
  .def("__repr__",&BatchedPtensors0b<float>::repr);


