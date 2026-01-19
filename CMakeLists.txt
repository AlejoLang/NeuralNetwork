cmake_minimum_required(VERSION 4.2)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

find_package(PkgConfig REQUIRED)
pkg_check_modules(MATIO REQUIRED matio)
find_package(OpenMP REQUIRED)

add_subdirectory(src)
add_subdirectory(vendor/SDL EXCLUDE_FROM_ALL)

target_include_directories(${PROJECT_NAME} PRIVATE ${MATIO_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${MATIO_LIBRARIES})
target_compile_options(${PROJECT_NAME} PRIVATE ${MATIO_CFLAGS_OTHER})

target_link_libraries(${PROJECT_NAME} PRIVATE 
    OpenMP::OpenMP_CXX
    SDL3::SDL3
)