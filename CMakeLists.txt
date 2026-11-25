cmake_minimum_required(VERSION 3.16)

project(stick_balance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLUT REQUIRED)

add_library(
    stick_balance_core
    src/cart_pole.cpp
)

target_include_directories(
    stick_balance_core
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(
    stick_balance
    src/viewer.cpp
)

target_link_libraries(
    stick_balance
    PRIVATE
    stick_balance_core
    glfw
    OpenGL::GL
    GLUT::GLUT
)

# Importable Python module sharing the original physics and GLFW drawing.
set_target_properties(stick_balance_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(stick_physics src/python_bridge.cpp src/viewer.cpp)
target_compile_definitions(stick_physics PRIVATE STICK_VIEWER_LIBRARY)
target_link_libraries(stick_physics PRIVATE stick_balance_core glfw OpenGL::GL GLUT::GLUT)
set_target_properties(stick_physics PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}"
)
