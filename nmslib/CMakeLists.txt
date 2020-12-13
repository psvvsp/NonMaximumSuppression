cmake_minimum_required (VERSION 3.8)

project(nmslib LANGUAGES CXX CUDA)

include(CTest)

set(SOURCE_FILES
    "nms.cpp"
    "nms_gpu.cu"
    "nms_single_thread.cpp"
    "nms_multiple_threads.cpp"
    "nms_impl.cpp"
    "box_generator.cpp")

set(HEADER_FILES 
    "nms.h"
    "box.h"
    "box_generator"
    "common.h"
    "nms_gpu.h"
    "nms_single_thread.h"
    "nms_multiple_threads.h")
 
add_library(nmslib STATIC ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties( nmslib
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(nmstest "main.cpp")
 
set_target_properties(nmstest 
             PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nmstest PRIVATE nmslib)
 