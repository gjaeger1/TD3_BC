cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TD3_BC_CPP)

find_package(Torch REQUIRED)

add_executable(td3_bc src/main.cpp)
target_link_libraries(td3_bc "${TORCH_LIBRARIES}")
set_property(TARGET td3_bc PROPERTY CXX_STANDARD 20)