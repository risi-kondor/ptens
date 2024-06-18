
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


/*
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

m.def("add_msg_n",[](Ptensors0& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_n",[](Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_n",[](Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);

m.def("add_msg",[](Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg",[](Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg",[](Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);

m.def("add_msg_n",[](Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_n",[](Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_n",[](Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);



m.def("add_msg_back",[](Ptensors0& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors1& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors2& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back_n",[](Ptensors1& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg_back_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back_n",[](Ptensors2& r, const Ptensors0& x, const Hgraph& G, int offs){return add_msg_back_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);

m.def("add_msg_back",[](Ptensors0& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back_n",[](Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_back_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back_n",[](Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs){return add_msg_back_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);

m.def("add_msg_back",[](Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back",[](Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_back(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back_n",[](Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_back_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);
m.def("add_msg_back_n",[](Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs){return add_msg_back_n(r,x,G,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("G"), py::arg("offs")=0);


m.def("add_msg_back",[](loose_ptr<Ptensors0>& r, const loose_ptr<Ptensors0>& x, const Hgraph& G){
    return add_msg_back(r,x,G.reverse());});
m.def("add_msg_back",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors0>& x, const Hgraph& G){
    return add_msg_back(r,x,G.reverse());});
m.def("add_msg_back",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors0>& x, const Hgraph& G){
    return add_msg_back(r,x,G.reverse());});
m.def("add_msg_back_n",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors0>& x, const Hgraph& G){
    return add_msg_back_n(r,x,G.reverse());});
m.def("add_msg_back_n",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors0>& x, const Hgraph& G){
    return add_msg_back_n(r,x,G.reverse());});

m.def("add_msg_back",[](loose_ptr<Ptensors0>& r, const loose_ptr<Ptensors1>& x, const Hgraph& G){
    return add_msg_back(r,x,G.reverse());});
m.def("add_msg_back",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors1>& x, const Hgraph& G){
    return add_msg_back(r,x,G.reverse());});
m.def("add_msg_back",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors1>& x, const Hgraph& G){
    return add_msg_back(r,x,G.reverse());});
m.def("add_msg_back_n",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors1>& x, const Hgraph& G){
    return add_msg_back_n(r,x,G.reverse());});
m.def("add_msg_back_n",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors1>& x, const Hgraph& G){
    return add_msg_back_n(r,x,G.reverse());});

m.def("add_msg_back",[](loose_ptr<Ptensors0>& r, const loose_ptr<Ptensors2>& x, const Hgraph& G){
    return add_msg_back(r,x,G.reverse());});
m.def("add_msg_back",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors2>& x, const Hgraph& G){
    return add_msg_back(r,x,G.reverse());});
m.def("add_msg_back",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors2>& x, const Hgraph& G){
    return add_msg_back(r,x,G.reverse());});
m.def("add_msg_back_n",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors2>& x, const Hgraph& G){
    return add_msg_back_n(r,x,G.reverse());});
m.def("add_msg_back_n",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors2>& x, const Hgraph& G){
    return add_msg_back_n(r,x,G.reverse());});


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


m.def("unite1",[](const Ptensors0& x, const Hgraph& G) {return unite1(x,G);});
m.def("unite2",[](const Ptensors0& x, const Hgraph& G) {return unite2(x,G);});
m.def("unite1",[](const Ptensors1& x, const Hgraph& G) {return unite1(x,G);});
m.def("unite2",[](const Ptensors1& x, const Hgraph& G) {return unite2(x,G);});
m.def("unite1",[](const Ptensors2& x, const Hgraph& G) {return unite1(x,G);});
m.def("unite2",[](const Ptensors2& x, const Hgraph& G) {return unite2(x,G);});
m.def("unite1_n",[](const Ptensors1& x, const Hgraph& G) {return unite1_n(x,G);});
m.def("unite2_n",[](const Ptensors1& x, const Hgraph& G) {return unite2_n(x,G);});
m.def("unite1_n",[](const Ptensors2& x, const Hgraph& G) {return unite1_n(x,G);});
m.def("unite2_n",[](const Ptensors2& x, const Hgraph& G) {return unite2_n(x,G);});
  
m.def("unite0to1_back",[](loose_ptr<Ptensors0>& x, loose_ptr<Ptensors1>& r, const Hgraph& G) {
    return add_msg_back(x,r,G.reverse());});
m.def("unite1to1_back",[](loose_ptr<Ptensors1>& x, loose_ptr<Ptensors1>& r, const Hgraph& G) {
    return add_msg_back(x,r,G.reverse());});
m.def("unite2to1_back",[](loose_ptr<Ptensors2>& x, loose_ptr<Ptensors1>& r, const Hgraph& G) {
    return add_msg_back(x,r,G.reverse());});
m.def("unite0to2_back",[](loose_ptr<Ptensors0>& x, loose_ptr<Ptensors2>& r, const Hgraph& G) {
    return add_msg_back(x,r,G.reverse());});
m.def("unite1to2_back",[](loose_ptr<Ptensors1>& x, loose_ptr<Ptensors2>& r, const Hgraph& G) {
    return add_msg_back(x,r,G.reverse());});
m.def("unite2to2_back",[](loose_ptr<Ptensors2>& x, loose_ptr<Ptensors2>& r, const Hgraph& G) {
    return add_msg_back(x,r,G.reverse());});

m.def("unite1to1_back_n",[](loose_ptr<Ptensors1>& x, loose_ptr<Ptensors1>& r, const Hgraph& G) {
    return add_msg_back_n(x,r,G.reverse());});
m.def("unite2to1_back_n",[](loose_ptr<Ptensors2>& x, loose_ptr<Ptensors1>& r, const Hgraph& G) {
    return add_msg_back_n(x,r,G.reverse());});
m.def("unite1to2_back_n",[](loose_ptr<Ptensors1>& x, loose_ptr<Ptensors2>& r, const Hgraph& G) {
    return add_msg_back_n(x,r,G.reverse());});
m.def("unite2to2_back_n",[](loose_ptr<Ptensors2>& x, loose_ptr<Ptensors2>& r, const Hgraph& G) {
    return add_msg_back_n(x,r,G.reverse());});


m.def("gather",[](const Ptensors0& x, const Hgraph& G) {return gather(x,G);});
m.def("gather_back",[](loose_ptr<Ptensors0>& x, loose_ptr<Ptensors0>& r, const Hgraph& G){
    add_gather(x,r,G.reverse());});
*/
