cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(mnist)

set(CMAKE_PREFIX_PATH /home/shiv/libtorch
                      /usr/include/opencv4)

set(CAFFE2_USE_CUDNN ON) # https://discuss.pytorch.org/t/use-cudnn-set-to-zero-after-upgrade-to-libtorch-2-0/175413/5
find_package(CUDAToolkit REQUIRED)  # Preferred over find_package(CUDA REQUIRED) https://cmake.org/cmake/help/latest/module/FindCUDA.html
option(USE_CUDNN "Enable cuDNN support" ON)

if(USE_CUDNN)
    set(CUDNN_ROOT_DIR "/usr/local/cuda" CACHE PATH "cuDNN root directory")
    
    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${CUDNN_ROOT_DIR}/include
        DOC "cuDNN include directory")
    
    find_library(CUDNN_LIBRARY cudnn
        HINTS ${CUDNN_ROOT_DIR}/lib64
        DOC "cuDNN library")
    
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
        add_definitions(-DUSE_CUDNN)
        set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
        set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
    else()
        message(WARNING "cuDNN not found. Compiling without cuDNN support.")
        set(USE_CUDNN OFF)
    endif()
endif()

# Include directories
include_directories(include)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Source files
set(SOURCES
    src/main.cpp
    src/Architecture.cpp
    src/Model.cpp
    src/Backbone.cpp
    src/utils.cpp
)

# Header files
set(HEADERS
    include/Architecture.hpp
    include/Model.hpp
    include/utils.hpp
    include/Backbone.hpp
    include/Trainer.hpp
    include/Trainer.tpp
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# Boost linking
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD 11)
