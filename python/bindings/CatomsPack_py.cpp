pybind11::class_<CompressedAtomsPack>(m,"catomspack")

  .def(py::init([](const AtomsPack& _atoms, at::Tensor& M){
	return CompressedAtomsPack(_atoms,tensorf(M));}))

  .def_static("random",[](const AtomsPack& _atoms, const int _nvecs){
      return CompressedAtomsPack(_atoms,_nvecs,4);})

  .def("__len__",&CompressedAtomsPack::size)
  .def("__eq__",&CompressedAtomsPack::operator==)

  .def("constk",[](const CompressedAtomsPack& x){return x.constk();})
  .def("nvecs",[](const CompressedAtomsPack& x){return x.nvecs();})
  .def("atoms",[](const CompressedAtomsPack& x){return x.atoms();})
  .def("atoms",[](const CompressedAtomsPack& x, const int i){return x.atoms(i);})
  .def("basis",[](const CompressedAtomsPack& x, const int i){return x.basis(i).torch();})

  .def("torch",[](const CompressedAtomsPack& x){return x.as_matrix().torch();})

  .def("str",&CompressedAtomsPack::str,py::arg("indent")="")
  .def("__str__",&CompressedAtomsPack::str,py::arg("indent")="")
  .def("__repr__",&CompressedAtomsPack::str,py::arg("indent")="");






