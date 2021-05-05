PROJECT(DEVO_SPSG_CPP)

cmake_minimum_required(VERSION 3.16)


include(CheckCXXCompilerFlag)


CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)


if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


# SET(OpenCV_DIR /home/thomas/Documents/opencv-3.4.13/build)
SET(OpenCV_DIR /home/teddy/Dev/opencv-3.4.13/build)

#SET(EXECUTABLE_OUTPUT_PATH ".")
FIND_PACKAGE(realsense2 CONFIG REQUIRED)
FIND_PACKAGE(OpenCV 3)


#SET(ZED_LIBS /usr/local/zed /usr/local/cuda ${CUDA_CUDART_LIBRARY})
MESSAGE("Message test" ${EXECUTABLE_OUTPUT_PATH} ${CMAKE_THREAD_LIBS_INIT})

ADD_EXECUTABLE(main Examples/main.cpp)
TARGET_LINK_LIBRARIES(
    main
    ${OpenCV_LIBS}
)