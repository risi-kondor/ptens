pybind11::class_<ptens::BatchedGgraph>(m,"batched_ggraph")

  .def(pybind11::init<const vector<Ggraph>&>())
  .def(pybind11::init<const vector<int>&>())

  .def_static("edge_index",[](at::Tensor& M, vector<int>& indicators){
      //auto T=Ltensor<float>(ATview<float>(M));
      return BatchedGgraph::from_edge_list(Ltensor<int>(ATview<int>(M)),indicators);
    })

  .def("__len__",&BatchedGgraph::size)
  .def("__getitem__",[](const BatchedGgraph& x, const int i){return x[i];})

  .def("subgraphs",[](const BatchedGgraph& G, const Subgraph& H){
      return G.subgraphs(H);})

  .def("str",&BatchedGgraph::str,py::arg("indent")="")
  .def("__str__",&BatchedGgraph::str,py::arg("indent")="");


