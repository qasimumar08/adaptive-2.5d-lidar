# =============================================================================
# CMakeLists.txt — Foveated Grid Engine (C++ / HIP)
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================
cmake_minimum_required(VERSION 3.21)

include(CheckLanguage)
check_language(HIP)

if(CMAKE_HIP_COMPILER)
    project(foveated_grid_engine LANGUAGES CXX HIP)
    find_package(hip QUIET)
    set(HAS_HIP TRUE)
    message(STATUS "HIP compiler detected: ${CMAKE_HIP_COMPILER}")
else()
    project(foveated_grid_engine LANGUAGES CXX)
    set(HAS_HIP FALSE)
    message(STATUS "HIP compiler not found — building with host C++ fallback")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 QUIET)

# ---- Grid Engine Library (C++) ----
add_library(grid_engine STATIC
    src/foveated_grid.cpp
    src/zone_manager.cpp
)
target_include_directories(grid_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(Eigen3_FOUND)
    target_link_libraries(grid_engine PUBLIC Eigen3::Eigen)
endif()

# ---- HIP Kernel Library (AMD GPU / Host Fallback) ----
add_library(grid_hip_kernels STATIC
    src/hip_projection.hip
)
target_include_directories(grid_hip_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(HAS_HIP)
    target_link_libraries(grid_hip_kernels PUBLIC hip::device)
else()
    set_source_files_properties(src/hip_projection.hip PROPERTIES LANGUAGE CXX)
endif()

# ---- Combined Library ----
add_library(foveated_grid INTERFACE)
target_link_libraries(foveated_grid INTERFACE grid_engine grid_hip_kernels)

# ---- Python Bindings (pybind11) ----
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python")
    set(Python_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python" CACHE FILEPATH "" FORCE)
    set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python" CACHE FILEPATH "" FORCE)
    set(Python_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/include/python3.12" CACHE PATH "" FORCE)
    set(Python3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/include/python3.12" CACHE PATH "" FORCE)
    set(Python_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/libpython3.12.so" CACHE FILEPATH "" FORCE)
    set(Python3_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/libpython3.12.so" CACHE FILEPATH "" FORCE)
    set(PYBIND11_FINDPYTHON ON)
    include_directories(SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/include/python3.12")
    include_directories(SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/include")
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib")
endif()

find_package(pybind11 QUIET)
if(pybind11_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pybind/grid_bindings.cpp")
    pybind11_add_module(grid_py pybind/grid_bindings.cpp)
    target_link_libraries(grid_py PRIVATE grid_engine grid_hip_kernels)
    message(STATUS "pybind11 found — building Python bindings")
else()
    message(STATUS "pybind/grid_bindings.cpp not found — skipping Python bindings")
endif()

# ---- Tests (Google Test) ----
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/test_grid_engine.cpp")
        enable_testing()
        add_executable(test_grid_engine
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/test_grid_engine.cpp
        )
        target_link_libraries(test_grid_engine PRIVATE
            grid_engine grid_hip_kernels GTest::GTest GTest::Main
        )
        add_test(NAME GridEngineTests COMMAND test_grid_engine)
    endif()
endif()

# ---- Install ----
install(TARGETS grid_engine grid_hip_kernels
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
