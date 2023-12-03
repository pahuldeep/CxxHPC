cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(qtLinkCmake LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtLinkCmake hello.cpp hello.h hello.cu)

# CUDA settings
set(CUDA_DIR $ENV{CUDA_PATH}) # Path to CUDA toolkit install
set(SYSTEM_NAME "x64")         # Depending on your system either 'Win32', 'x64', or 'Win64'
set(SYSTEM_TYPE 64)            # '32' or '64', depending on your system
set(CUDA_ARCH "all")           # Type of CUDA architecture
set(NVCC_OPTIONS "--use_fast_math")

# include paths
include_directories(${CUDA_DIR}/include)

# The following makes sure all path names (which often include spaces) are put between quotation marks
set(CUDA_INC "-I${CUDA_DIR}/include")

link_directories(${CUDA_DIR}/lib/${SYSTEM_NAME})

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
set(MSVCRT_LINK_FLAG_DEBUG "/MDd")

# CUDA libraries
set(CUDA_LIBS
    cublas cublasLt cuda cudadevrt cudart cudart_static cudnn cudnn64_8 cudnn_adv_infer
    cudnn_adv_infer64_8 cudnn_adv_train cudnn_adv_train64_8 cudnn_cnn_infer cudnn_cnn_infer64_8
    cudnn_cnn_train cudnn_cnn_train64_8 cudnn_ops_infer cudnn_ops_infer64_8 cudnn_ops_train
    cudnn_ops_train64_8 cufft cufftw cufilt curand cusolver cusolverMg cusparse nppc nppial
    nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps nvblas nvjpeg nvml
    nvptxcompiler_static nvrtc-builtins_static nvrtc nvrtc_static OpenCL
)

# Add library dependencies
foreach(lib ${CUDA_LIBS})
    set(CUDA_LIB "${CUDA_LIB} -l${lib}")
endforeach()
set(LIBS "${LIBS} ${CUDA_LIB}")

# Debug mode
set(CUDA_SOURCES hello.cu)  # Add your CUDA source files here
set(CUDA_OBJECTS_DIR .)
set(CUDA_OUTPUT ${CUDA_OBJECTS_DIR}/${CMAKE_FILE_BASE}_cuda.obj)

add_custom_target(
    cuda_target
    COMMAND ${CUDA_DIR}/bin/nvcc.exe -D_DEBUG ${NVCC_OPTIONS} ${CUDA_INC} ${LIBS}
            --machine ${SYSTEM_TYPE} -arch=${CUDA_ARCH}
            --compile -cudart static -g -DWIN32 -D_MBCS
            -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1"
            -Xcompiler ${MSVCRT_LINK_FLAG_DEBUG}
            -c -o ${CUDA_OUTPUT} ${CUDA_SOURCES}
    DEPENDS ${CUDA_SOURCES}
)


include(GNUInstallDirs)
install(TARGETS qtLinkCmake
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
