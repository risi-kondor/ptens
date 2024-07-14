pybind11::class_<cnine::SimpleMemoryManager>(m,"vram_manager")

  .def_static("reset",[](const int s){
      if(ptens_global::vram_manager) delete ptens_global::vram_manager;
      ptens_global::vram_manager=new SimpleMemoryManager(cnine::Mbytes(s),1);
    })

  .def_static("size",[](){
      return ptens_global::vram_manager->size();})

  .def_static("clear",[](){
      if(ptens_global::vram_manager) ptens_global::vram_manager->clear();});
