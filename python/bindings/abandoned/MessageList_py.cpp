pybind11::class_<MessageList>(m,"message_list")

  .def("str",[](const MessageList& x){return x.str();})
  .def("__str__",[](const MessageList& x){return x.str();})
  .def("__repr__",[](const MessageList& x){return x.repr();});
