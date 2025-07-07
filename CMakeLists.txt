cmake_minimum_required(VERSION 3.10.2)

# Set version variables
set(version_major "0")
set(version_minor "0")
set(version_patch "1")

# Set project version
set(project_version "${version_major}.${version_minor}.${version_patch}")

# Set the project and description
project(ProjectName VERSION ${project_version} DESCRIPTION "cpp data modelling" LANGUAGES CXX)

# Set custom cmake modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Set output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# Use c++11 flag for all modules
add_definitions("-std=c++17")

# Conan setup for dependencies
# include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

# Add sources
add_subdirectory(src)


# Testing
option(CPPDATAMODELLING_BUILD_TESTS "Determines whether to build tests." ON)
if(CPPDATAMODELLING_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif(CPPDATAMODELLING_BUILD_TESTS)
