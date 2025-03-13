set(ptens_cuda_source ${cnine_cuda_source} Ptensors0.cu Ptensors1.cu Ptensors2.cu)
set_source_files_properties(${ptens_cuda_source} PROPERTIES LANGUAGE CUDA)

target_sources(ptens PRIVATE ${ptens_cuda_source})
target_compile_options(ptens PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)

set_target_properties(ptens PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ptens PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)



