cmake_minimum_required(VERSION 3.7)

file(GLOB COMPILER_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

# PYTORCH_DIR
IF(DEFINED ENV{PYTORCH_DIR})
  SET(PYTORCH_DIR $ENV{PYTORCH_DIR})
ENDIF()

IF ("${PYTORCH_DIR}" STREQUAL "")
  message(FATAL_ERROR "Please specify the PyTorch directory with -DPYTORCH_DIR=/path/to/pytorch/dir")
ENDIF()

# JULIA_DIR
IF(DEFINED ENV{JULIA_DIR})
  SET(JULIA_DIR $ENV{JULIA_DIR})
ENDIF()

execute_process(COMMAND "${JULIA_DIR}/julia-config.jl" "--cflags" OUTPUT_VARIABLE JULIA_CFLAGS) 
execute_process(COMMAND "${JULIA_DIR}/julia-config.jl" "--ldflags" OUTPUT_VARIABLE JULIA_LDFLAGS) 
execute_process(COMMAND "${JULIA_DIR}/julia-config.jl" "--ldlibs" OUTPUT_VARIABLE JULIA_LIBS) 
string(STRIP ${JULIA_CFLAGS} JULIA_CFLAGS)
string(STRIP ${JULIA_LDFLAGS} JULIA_LDFLAGS)
string(STRIP ${JULIA_LIBS} JULIA_LIBS)
message("RUNNING ${JULIA_DIR}/julia-config.jl --cflags")
message("GOT SOME CFLAGS ${JULIA_CFLAGS}")
message("GOT SOME LDFLAGS ${JULIA_LDFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${JULIA_CFLAGS}")

IF ("${JULIA_DIR}" STREQUAL "")
  message(FATAL_ERROR "Please specify the Julia directory with -DJULIA_DIR=/path/to/julia/share/dir")
ENDIF()

message("Using PyTorch directory ${PYTORCH_DIR}")
message("Using Julia directory ${JULIA_DIR}")

link_directories(${PYTORCH_DIR}/lib)

add_subdirectory(pybind11)

pybind11_add_module(torch_julia SHARED ${COMPILER_SRCS})
target_link_libraries(torch_julia PUBLIC torch pybind11 ${JULIA_LDFLAGS} ${JULIA_LIBS})

target_include_directories(torch_julia PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PYTORCH_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
)
