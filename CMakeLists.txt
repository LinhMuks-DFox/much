cmake_minimum_required(VERSION 3.10)
project(much C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

find_package(BLAS REQUIRED)

add_executable(much main.c tensor.c layer.c sequence.c util.c mse.c crossentropy.c mnist.c argmax.c optimizer.c)

target_include_directories(much PRIVATE ${BLAS_INCLUDE_DIRS})
target_link_libraries(much PRIVATE ${BLAS_LIBRARIES} m)

