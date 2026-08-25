# Find the oneCCL (XCCL) runtime used by the c10d XCCL backends.
#
# This will define the following variables:
#   XCCL_FOUND        : True if the system has the XCCL library.
#   XCCL_INCLUDE_DIR  : Include directories needed to use XCCL.
#   XCCL_LIBRARY_DIR  : The path to the XCCL library directory.
#   XCCL_LIBRARY      : libccl.so.1, the C++ `ccl::` API.
#   XCCL_LIBRARY_2_0  : libccl.so.2.0, the `onecclXxx` C API used by xccl2.
#   XCCL_VERSION      : ONECCL_MAJOR.ONECCL_MINOR.ONECCL_PATCH from ccl.h.
#
# oneCCL ships the two APIs as separate sonames out of one install tree, so both
# libraries are required: the torch-xpu-ops backend links the C++ one and the
# xccl2 backend links the C one.

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(XCCL_FOUND False)
  set(XCCL_NOT_FOUND_MESSAGE "OneCCL library is only supported on Linux!")
  return()
endif()

# we need source OneCCL environment before building.
set(XCCL_ROOT $ENV{CCL_ROOT})

find_file(
  XCCL_INCLUDE_DIR
  NAMES include
  HINTS ${XCCL_ROOT}
  NO_DEFAULT_PATH
)

find_file(
  XCCL_INCLUDE_ONEAPI_DIR
  NAMES oneapi
  HINTS ${XCCL_ROOT}/include/
  NO_DEFAULT_PATH
)

find_file(
  XCCL_LIBRARY_DIR
  NAMES lib
  HINTS ${XCCL_ROOT}
  NO_DEFAULT_PATH
)

find_library(
  XCCL_LIBRARY
  NAMES libccl.so.1
  HINTS ${XCCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

find_file(
  XCCL_LIBRARY_2_0
  NAMES libccl.so.2.0
  HINTS ${XCCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

if((NOT XCCL_INCLUDE_DIR) OR (NOT XCCL_INCLUDE_ONEAPI_DIR) OR (NOT XCCL_LIBRARY_DIR) OR (NOT XCCL_LIBRARY) OR (NOT XCCL_LIBRARY_2_0))
  set(XCCL_FOUND False)
  set(XCCL_NOT_FOUND_MESSAGE "OneCCL library not found!!")
  return()
endif()

# ONECCL_MAJOR/MINOR/PATCH describe the C API; the older CCL_*_VERSION macros in
# oneapi/ccl/config.h describe the C++ one and are not interchangeable.
set(XCCL_VERSION "")
if(EXISTS "${XCCL_INCLUDE_ONEAPI_DIR}/ccl.h")
  file(STRINGS "${XCCL_INCLUDE_ONEAPI_DIR}/ccl.h" XCCL_VERSION_DEFINES
       REGEX "^#define ONECCL_(MAJOR|MINOR|PATCH) ")
  foreach(_component MAJOR MINOR PATCH)
    string(REGEX MATCH "#define ONECCL_${_component} ([0-9]+)"
           _matched "${XCCL_VERSION_DEFINES}")
    if(_matched)
      set(XCCL_VERSION_${_component} ${CMAKE_MATCH_1})
    endif()
  endforeach()
  if(DEFINED XCCL_VERSION_MAJOR AND DEFINED XCCL_VERSION_MINOR AND DEFINED XCCL_VERSION_PATCH)
    set(XCCL_VERSION "${XCCL_VERSION_MAJOR}.${XCCL_VERSION_MINOR}.${XCCL_VERSION_PATCH}")
  endif()
endif()

list(APPEND XCCL_INCLUDE_DIR ${XCCL_INCLUDE_ONEAPI_DIR})

SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
  "${XCCL_INCLUDE_DIR}")
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
  "${XCCL_LIBRARY_DIR}")

find_package_handle_standard_args(
  XCCL
  FOUND_VAR XCCL_FOUND
  REQUIRED_VARS XCCL_INCLUDE_DIR XCCL_LIBRARY_DIR XCCL_LIBRARY XCCL_LIBRARY_2_0
  VERSION_VAR XCCL_VERSION
  REASON_FAILURE_MESSAGE "${XCCL_NOT_FOUND_MESSAGE}"
)
