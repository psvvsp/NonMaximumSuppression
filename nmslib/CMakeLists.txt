cmake_minimum_required (VERSION 3.8)

project(nmslib LANGUAGES CXX CUDA)

include(CTest)

set(HEADERS_DIR ../headers)

set(SOURCE_FILES
    "nms.cpp"
    "nms_gpu.cu"
    "nms_single_thread.cpp"
    "nms_multiple_threads.cpp"
    "nms_impl.cpp"
    "box_generator.cpp")

set(HEADER_FILES 
    "${HEADERS_DIR}/nms.h"
    "${HEADERS_DIR}/box.h"
    "${HEADERS_DIR}/common.h"
    "${HEADERS_DIR}/nms_interface.h"
    "box_generator"
    "nms_gpu.h"
    "nms_impl.h"
    "nms_single_thread.h"
    "nms_multiple_threads.h")

add_library(nmslib STATIC ${HEADER_FILES} ${SOURCE_FILES})
 
target_include_directories(nmslib PRIVATE ${HEADERS_DIR})

set_target_properties( nmslib
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(nmstest "test.cpp")

target_include_directories(nmstest PRIVATE ${HEADERS_DIR})
 
set_target_properties(nmstest 
             PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nmstest PRIVATE nmslib)