set(ptens_cuda_source
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Ptensors0.cu>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Ptensors1.cu>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Ptensors2.cu>
)
set_source_files_properties(${ptens_cuda_source} PROPERTIES LANGUAGE CUDA)

target_sources(ptens PRIVATE ${ptens_cuda_source})
target_compile_options(ptens PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
target_link_libraries(ptens PRIVATE CUDA::cudart CUDA::cuda_driver)

set_target_properties(ptens PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ptens PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)



