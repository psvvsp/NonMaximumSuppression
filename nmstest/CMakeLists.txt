cmake_minimum_required (VERSION 3.8)
 
project(nmstest LANGUAGES CXX CUDA)
 
set(SOURCE_FILES "main.cpp")
set(HEADER_FILES "box_generator.h")
 
add_executable (nmstest ${SOURCE_FILES} ${HEADER_FILES})
 
include_directories("../nmslib")
target_link_libraries(nmstest nmslib)