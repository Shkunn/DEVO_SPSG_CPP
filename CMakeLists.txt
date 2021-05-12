project(DEVO_SPSG_CPP)

cmake_minimum_required(VERSION 3.16)

IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")


# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# SET(OpenCV_DIR /home/thomas/Documents/opencv-3.4.13/build)
SET(OpenCV_DIR /home/teddy/Dev/opencv/build)
SET(Torch_DIR /home/teddy/Dev/libtorch/share/cmake/Torch)  # libtorch

#SET(EXECUTABLE_OUTPUT_PATH ".")
FIND_PACKAGE(realsense2 CONFIG REQUIRED)
FIND_PACKAGE(OpenCV 4)
FIND_PACKAGE(Torch REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/SPextractor.cc
src/SuperPoint.cc
)

target_link_libraries(${PROJECT_NAME}
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)

ADD_EXECUTABLE(main Examples/main.cpp)
TARGET_LINK_LIBRARIES(
    main
    ${PROJECT_NAME}
)