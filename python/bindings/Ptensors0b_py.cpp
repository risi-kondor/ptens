//pybind11::class_<cnine::RtensorPack>(m,"rtensor_pack");

pybind11::class_<Ptensors0b<float> >(m,"ptensors0b")

  .def(pybind11::init<const at::Tensor&>())
  .def(pybind11::init<const at::Tensor&, const AtomsPack&>())
  .def(pybind11::init<const at::Tensor&, const vector<vector<int> >& >())

  .def_static("create",[](const int n, const int _nc, const int fcode, const int _dev){
      return Ptensors0b<float>(n,_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const vector<vector<int> > _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors0b<float>(AtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const AtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors0b<float>(_atoms,cnine::channels=_nc,cnine::filltype=fcode,cnine::device=_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def("copy",&Ptensors0b<float>::copy)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](Ptensors0b<float>& x, const at::Tensor& y){x.add_to_grad(ATview<float>(y));})
  .def("add_to_grad",[](Ptensors0b<float>& x, const Ptensors0b<float>& y, const float c){x.add_to_grad(y,c);})

//.def("add_to_grad",[](Ptensors0b<float>& x, const cnine::loose_ptr<Ptensors0b<float>>& y){x.add_to_grad(y);})
//  .def("add_to_gradp",[](Ptensors0b<float>& x, const cnine::loose_ptr<Ptensors0b<float>>& y){x.add_to_grad(y);})
//.def("add_to_grad",&Ptensors0b<float>::add_to_grad)
//.def("get_grad",&Ptensors0b<float>::get_grad)
//.def("get_gradp",&Ptensors0b<float>::get_gradp)
//.def("gradp",&Ptensors0b<float>::get_gradp)

//.def("add_to_grad",[](Ptensors0b<float>& x, const int i, at::Tensor& T){
//x.get_grad().view_of_tensor(i).add(RtensorA::view(T));
//    })

//  .def("view_of_grad",&Ptensors0b<float>::view_of_grad)

  .def("__getitem__",[](const Ptensors0b<float>& x, const int i){return x(i);})
  .def("torch",[](const Ptensors0b<float>& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&Ptensors0b<float>::get_dev)
  .def("get_nc",&Ptensors0b<float>::get_nc)
//.def("get_atoms",[](const Ptensors0b<float>& x){return x.atoms;})
//.def("view_of_atoms",[](const Ptensors0b<float>& x){return x.atoms;})
//.def("view_of_atoms",&Ptensors0b<float>::view_of_atoms)


//.def("atoms_of",[](const Ptensors0b<float>& x, const int i){return vector<int>(x.atoms_of(i));})
   //  .def("push_back",&Ptensors0b<float>::push_back)

  .def("to_device",&Ptensors0b<float>::move_to_device)
   //.def("move_to_device_back",[](Ptensors0b<float>& x, const cnine::loose_ptr<Ptensors0b<float>>& g, const int dev){
   // if(!x.grad) x.grad=new Ptensors0b<float>(g,dev);
   // else x.grad->add(Ptensors0b<float>(g,dev));})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",&Ptensors0b::add)

  .def("mprod",[](const Ptensors0b<float>& x, at::Tensor& M){
      return mprod(x,ATview<float>(M));})
  .def("add_mprod_back0",[](Ptensors0b<float>& r, const Ptensors0b<float>& g, at::Tensor& M){
      r.add_mprod_back0(g,ATview<float>(M));})
  .def("mprod_back1",[](const Ptensors0b<float>& x, const Ptensors0b<float>& g){
      return (x.transp()*g.get_grad()).torch();})

  .def("scale_channels",[](Ptensors0b<float>& x, at::Tensor& y){
      return scale_channels(x,ATview<float>(y));})
  .def("add_scale_channels_back0",[](Ptensors0b<float>& r, const Ptensors0b<float>& g, at::Tensor& y){
      return r.add_scale_channels_back(g,ATview<float>(y));})

  .def("linear",[](const Ptensors0b<float>& x, at::Tensor& y, at::Tensor& b){
      return linear(x,ATview<float>(y),ATview<float>(b));})
  .def("add_linear_back0",[](Ptensors0b<float>& r, const Ptensors0b<float>& g, at::Tensor& y){
      r.add_linear_back0(g,ATview<float>(y));})
  .def("linear_back1",[](const Ptensors0b<float>& x, const Ptensors0b<float>& g){
      return (x.transp()*g.get_grad()).torch();})
  .def("linear_back2",[](const Ptensors0b<float>& x, Ptensors0b<float>& g){
      return g.get_grad().sum(0).torch();})

  .def("ReLU",[](const Ptensors0b<float>& x, const float alpha){
      return ReLU(x,alpha);})
  .def("add_ReLU_back",&Ptensors0b<float>::add_ReLU_back)

  .def("inp",[](const Ptensors0b<float>& x, const Ptensors0b<float>& y){return x.inp(y);})
  .def("diff2",[](const Ptensors0b<float>& x, const Ptensors0b<float>& y){return x.diff2(y);})

// ---- I/O --------------------------------------------------------------------------------------------------

  .def("str",&Ptensors0b<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensors0b<float>::str,py::arg("indent")="")
  .def("__repr__",&Ptensors0b<float>::repr);


//pybind11::class_<loose_ptr<Ptensors0b<float>> >(m,"ptensors0_lptr");

//.def("add_mprod",&Ptensors0b<float>::_add_mprod)
//.def("add_mprod_back0",&Ptensors0b<float>::add_mprod_back0)
//.def("add_mprod_back1",&Ptensors0b<float>::add_mprod)

   /*
  .def("add_scale",[](Ptensors0b<float>& r, const Ptensors0b<float>& x, at::Tensor& y){
      RtensorA Y(y);
      Y.move_to_device(0);
      PTENS_ASSRT(Y.ndims()==1);
      PTENS_ASSRT(Y.dims[0]==1);
      r.add(x,Y(0));})
  .def("add_scale_back0",[](Ptensors0b<float>& x, const cnine::loose_ptr<Ptensors0b<float>>& g, at::Tensor& y){
      RtensorA Y(y);
      Y.move_to_device(0);
      PTENS_ASSRT(Y.ndims()==1);
      PTENS_ASSRT(Y.dims[0]==1);
      x.get_grad().add(g,Y(0));})
  .def("scale_back1",[](Ptensors0b<float>&x, const cnine::loose_ptr<Ptensors0b<float>>& g){
      RtensorA R(Gdims(1));
      R.set(0,x.inp(*g));
      return R.move_to_device(g->dev).torch();})
   */

//.def("average",&Ptensors0b<float>::average)
//.def("average_back",[](Ptensors0b<float>& x, Ptensors0b<float>& r){
//      x.get_grad().add_average_back(r.get_grad());})

