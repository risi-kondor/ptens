pybind11::class_<SGlayer1f>(m,"sglayer1")

  .def_static("view",[](const Ggraph& G, const Subgraph& S, const AtomsPack& atoms, at::Tensor& x){
      return SGlayer1f(G,S,atoms,tensorf::view(x));})

  .def("add_schur",[](SGlayer1f& r, const SGlayer1f& x, at::Tensor& W, at::Tensor& B){
      return r.add_schur(x,tensorf::view(W),tensorf::view(B));})
  .def("add_schur_back0",[](SGlayer1f& x, SGlayer1f& r, at::Tensor& W){
      x.add_schur_back0(r,tensorf::view(W));})
  .def("schur_back1",[](SGlayer1f& x, at::Tensor& W, at::Tensor& B, SGlayer1f& r){
      x.add_schur_back1_to(tensorf::view(W),tensorf::view(B),r);});

