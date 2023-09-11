pybind11::class_<Subgraph>(m,"subgraph")

//.def(pybind11::init<const vector<vector<int> >& >())

  .def_static("trivial",&Subgraph::trivial)
  .def_static("edge",&Subgraph::edge)
  .def_static("triangle",&Subgraph::triangle)
  .def_static("cycle",&Subgraph::cycle)
  .def_static("star",&Subgraph::star)

  .def_static("edge_index",[](const at::Tensor& x, int n=-1){
      return Subgraph::edge_index(cnine::RtensorA(x),n);})
    //.def_static("edge_index",[](const at::Tensor& x, const int n, const int m){
    //return Subgraph::edge_index(cnine::RtensorA(x),n,m);})
    //.def_static("edge_index",[](const at::Tensor& x, const at::Tensor& l, const int n){
    //return Subgraph::edge_index(cnine::RtensorA(x),cnine::RtensorA(l),n);})

  .def_static("matrix",[](const at::Tensor& x){return Subgraph(cnine::RtensorA(x));})
//.def_static("matrix",[](const at::Tensor& x, const at::Tensor& l){
//    return Subgraph(cnine::RtensorA(x),cnine::RtensorA(l));})

  .def("dense",[](const Subgraph& G){return G.dense().torch();})

  .def("str",&Subgraph::str,py::arg("indent")="")
  .def("__str__",&Subgraph::str,py::arg("indent")="");


