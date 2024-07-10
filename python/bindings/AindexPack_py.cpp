pybind11::class_<AindexPack>(m,"aindexpack")

  .def_static("from_lists",[](unordered_map<int,vector<int> >& x){
      AindexPack R;
      for(auto& p:x) 
	R.push_back(p.first,p.second);
      return R;})

  .def("target",[](const AindexPack& obj, const int i){return obj.tix(i);})
  .def("nindices",[](const AindexPack& obj, const int i){return obj.nix(i);})
  .def("indices",[](const AindexPack& obj, const int i){return obj.ix(i);})

  .def("push_back",[](AindexPack& obj, const int tix, vector<int> indices){obj.push_back(tix,indices);})
  
  .def("__str__",[](const AindexPack& obj){return obj.str();}); 



