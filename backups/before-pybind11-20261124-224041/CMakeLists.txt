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

# The same physics core is linked into the Python shared libraries.
set_target_properties(stick_balance_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(stick_rl SHARED src/python_bridge.cpp)
target_link_libraries(stick_rl PRIVATE stick_balance_core)

add_library(stick_viewer SHARED src/viewer.cpp)
target_compile_definitions(stick_viewer PRIVATE STICK_VIEWER_LIBRARY)
target_link_libraries(stick_viewer PRIVATE stick_balance_core glfw OpenGL::GL GLUT::GLUT)

# Keep the paths loaded by Python independent of the build configuration.
set_target_properties(stick_rl stick_viewer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}"
)
