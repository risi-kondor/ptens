pybind11::class_<MessageMap>(m,"message_map")

  .def("str",[](const MessageMap& x){return x.str();})
  .def("__str__",[](const MessageMap& x){return x.str();})
  .def("__repr__",[](const MessageMap& x){return x.repr();});

