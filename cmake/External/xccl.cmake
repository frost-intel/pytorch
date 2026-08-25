# Defines the torch::xccl interface target (oneCCL headers + libccl).
#
# third_party/torch-xpu-ops carries a copy of this logic for its own standalone
# builds. Both files share the __XCCL_INCLUDED guard, and this one runs first
# (from Dependencies.cmake, before the torch-xpu-ops subdirectory is added), so
# the in-tree definition wins and the version floor below is what is enforced.
if(NOT __XCCL_INCLUDED)
  set(__XCCL_INCLUDED TRUE)

  # Floor: the oldest oneCCL verified to build the xccl2 backend. The
  # onecclXxx C API that XcclApi wraps is not present in older releases, and
  # its coverage still varies between them (onecclCommAbort, for one, is
  # declared but not implemented), so raise this only with evidence.
  set(TORCH_XCCL_MINIMUM_VERSION 2021.17)

  # XCCL_ROOT, XCCL_LIBRARY_DIR, XCCL_INCLUDE_DIR are handled by FindXCCL.cmake.
  find_package(XCCL ${TORCH_XCCL_MINIMUM_VERSION})
  if(NOT XCCL_FOUND)
    set(PYTORCH_FOUND_XCCL FALSE)
    message(WARNING "${XCCL_NOT_FOUND_MESSAGE}")
  else()
    set(PYTORCH_FOUND_XCCL TRUE)
    if(NOT TARGET torch::xccl)
      add_library(torch::xccl INTERFACE IMPORTED)
      set_property(
        TARGET torch::xccl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${XCCL_INCLUDE_DIR})
      # oneCCL declares every onecclXxx entry point __attribute__((weak)), so a
      # reference to one does not mark libccl.so.2 as used: --as-needed drops it
      # from DT_NEEDED, the calls bind to NULL, and the first collective
      # segfaults rather than the library failing to load. libccl.so.1 needs no
      # wrapper, since the C++ ccl:: references to it are strong. Drop this once
      # the weak attribute goes away upstream.
      set_property(
        TARGET torch::xccl PROPERTY INTERFACE_LINK_LIBRARIES
        ${XCCL_LIBRARY}
        "-Wl,--no-as-needed,${XCCL_LIBRARY_2_0},--as-needed")
    endif()
  endif()
endif()
