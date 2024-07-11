pybind11::class_<ptens::OverlapsMmapCache>(m,"overlaps_tmap_cache")

  .def_static("enable",[](const bool x){ptens_global::cache_overlap_maps=x;})

  .def_static("size",[](){return ptens_global::overlaps_cache.size();});

