cmake_minimum_required(VERSION 3.10)
project(much C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-O3")

find_package(BLAS REQUIRED)

set(MUCH_IMPL_SOURCES
  impl/argmax.c
  impl/crossentropy.c
  impl/layer.c
  impl/mnist.c
  impl/mse.c
  impl/optimizer.c
  impl/sequence.c
  impl/tensor.c
  impl/util.c
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(much_core STATIC ${MUCH_IMPL_SOURCES})

target_include_directories(much_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BLAS_INCLUDE_DIRS}
)

if(NOT APPLE)
  target_link_libraries(much_core PUBLIC m)
endif()

target_link_libraries(much_core PUBLIC ${BLAS_LIBRARIES})

add_executable(much src/main.c)

target_link_libraries(much PRIVATE much_core)
