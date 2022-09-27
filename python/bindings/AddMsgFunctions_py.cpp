
m.def("add_msg",[](Ptensor0& r, const Ptensor0& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor1& r, const Ptensor0& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor2& r, const Ptensor0& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_msg",[](Ptensor0& r, const Ptensor1& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor1& r, const Ptensor1& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor2& r, const Ptensor1& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_msg",[](Ptensor0& r, const Ptensor2& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor1& r, const Ptensor2& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor2& r, const Ptensor2& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);


m.def("add_msg_back",[](Ptensor0& r, const Ptensor0& x, int offs){return add_msg_back(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensor1& r, const Ptensor0& x, int offs){return add_msg_back(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensor2& r, const Ptensor0& x, int offs){return add_msg_back(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_msg_back",[](Ptensor0& r, const Ptensor1& x, int offs){return add_msg_back(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensor1& r, const Ptensor1& x, int offs){return add_msg_back(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensor2& r, const Ptensor1& x, int offs){return add_msg_back(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_msg_back",[](Ptensor0& r, const Ptensor2& x, int offs){return add_msg_back(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensor1& r, const Ptensor2& x, int offs){return add_msg_back(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensor2& r, const Ptensor2& x, int offs){return add_msg_back(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);



m.def("add_msg",[](Ptensors0& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg",[](Ptensors1& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg",[](Ptensors2& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);

m.def("add_msg",[](Ptensors0& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg",[](Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg",[](Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);

m.def("add_msg",[](Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg",[](Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg",[](Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);


m.def("add_msg_back",[](Ptensors0& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors1& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors2& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);

m.def("add_msg_back",[](Ptensors0& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);

m.def("add_msg_back",[](Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);



m.def("msg_layer0",[](Ptensors0& x, const AtomsPack& atoms, const Hgraph& G){
    Ptensors0 R=Ptensors0::zero(atoms,x.nc,x.dev);
    add_msg(R,x,G); return R;});
m.def("msg_layer0",[](Ptensors1& x, const AtomsPack& atoms, const Hgraph& G){
    Ptensors0 R=Ptensors0::zero(atoms,x.nc,x.dev);
    add_msg(R,x,G); return R;});
m.def("msg_layer0",[](Ptensors2& x, const AtomsPack& atoms, const Hgraph& G){
    Ptensors0 R=Ptensors0::zero(atoms,2*x.nc,x.dev);
    add_msg(R,x,G); return R;});

m.def("msg_layer1",[](Ptensors0& x, const AtomsPack& atoms, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(atoms,x.nc,x.dev);
    add_msg(R,x,G); return R;});
m.def("msg_layer1",[](Ptensors1& x, const AtomsPack& atoms, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(atoms,2*x.nc,x.dev);
    add_msg(R,x,G); return R;});
m.def("msg_layer1",[](Ptensors2& x, const AtomsPack& atoms, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(atoms,5*x.nc,x.dev);
    add_msg(R,x,G); return R;});

m.def("msg_layer2",[](Ptensors0& x, const AtomsPack& atoms, const Hgraph& G){
    Ptensors2 R=Ptensors2::zero(atoms,2*x.nc,x.dev);
    add_msg(R,x,G); return R;});
m.def("msg_layer2",[](Ptensors1& x, const AtomsPack& atoms, const Hgraph& G){
    Ptensors2 R=Ptensors2::zero(atoms,5*x.nc,x.dev);
    add_msg(R,x,G); return R;});
m.def("msg_layer2",[](Ptensors2& x, const AtomsPack& atoms, const Hgraph& G){
    Ptensors2 R=Ptensors2::zero(atoms,15*x.nc,x.dev);
    add_msg(R,x,G); return R;});



