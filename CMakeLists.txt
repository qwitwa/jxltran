# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# jxltran builds standalone (cmake -S . -B build) or as part of
# libjxl via add_subdirectory from tools/CMakeLists.txt.
#
# When embedded, the parent must use add_subdirectory(jxltran <binary_dir>)
# with a binary_dir other than "jxltran" (e.g. jxltran_build): otherwise
# CMake's build folder .../tools/jxltran/ collides with the runtime output
# path .../tools/jxltran (the executable next to cjxl/djxl).

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(jxltran LANGUAGES CXX)
  set(JXLTRAN_STANDALONE ON)
endif()

set(JXLTRAN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(JXLTRAN_SOURCES
  "${JXLTRAN_SRC_DIR}/box.cc"
  "${JXLTRAN_SRC_DIR}/cmdline.cc"
  "${JXLTRAN_SRC_DIR}/codestream.cc"
  "${JXLTRAN_SRC_DIR}/compressed_icc.cc"
  "${JXLTRAN_SRC_DIR}/entropy.cc"
  "${JXLTRAN_SRC_DIR}/frame_header.cc"
  "${JXLTRAN_SRC_DIR}/image_header.cc"
  "${JXLTRAN_SRC_DIR}/lfglobal.cc"
  "${JXLTRAN_SRC_DIR}/jxltran.cc"
  "${JXLTRAN_SRC_DIR}/orientation_compose.cc"
  "${JXLTRAN_SRC_DIR}/photon_noise.cc"
  "${JXLTRAN_SRC_DIR}/operations.cc"
  "${JXLTRAN_SRC_DIR}/opsin_adjust.cc"
  "${JXLTRAN_SRC_DIR}/restoration_filter.cc"
  "${JXLTRAN_SRC_DIR}/spline_io.cc"
  "${JXLTRAN_SRC_DIR}/toc.cc"
  "${JXLTRAN_SRC_DIR}/toc_group_order.cc"
  "${JXLTRAN_SRC_DIR}/toc_layout.cc"
  "${JXLTRAN_SRC_DIR}/trace.cc"
)

add_executable(jxltran ${JXLTRAN_SOURCES})

set_property(TARGET jxltran PROPERTY CXX_STANDARD 17)
set_property(TARGET jxltran PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET jxltran PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(jxltran PRIVATE "${JXLTRAN_SRC_DIR}")

option(JXLTRAN_ENABLE_BROTLI
  "Enable brob compression/decompression in jxltran (requires brotli)" ON)

if(JXLTRAN_STANDALONE)
  set_target_properties(jxltran PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  if(JXLTRAN_ENABLE_BROTLI)
    find_path(JXLTRAN_BROTLI_INCLUDE NAMES brotli/encode.h)
    find_library(JXLTRAN_BROTLIENC NAMES brotlienc)
    find_library(JXLTRAN_BROTLIDEC NAMES brotlidec)
    if(JXLTRAN_BROTLI_INCLUDE AND JXLTRAN_BROTLIENC AND JXLTRAN_BROTLIDEC)
      target_compile_definitions(jxltran PRIVATE JXLTRAN_HAVE_BROTLI)
      target_include_directories(jxltran PRIVATE ${JXLTRAN_BROTLI_INCLUDE})
      target_link_libraries(jxltran PRIVATE
        ${JXLTRAN_BROTLIENC} ${JXLTRAN_BROTLIDEC})
      message(STATUS "jxltran: brob support enabled (system brotli)")
    else()
      message(STATUS "jxltran: brotli not found, brob support disabled")
    endif()
  endif()
else()
  set_target_properties(jxltran PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  if(JXLTRAN_ENABLE_BROTLI)
    if(TARGET brotlienc AND TARGET brotlidec)
      target_compile_definitions(jxltran PRIVATE JXLTRAN_HAVE_BROTLI)
      target_link_libraries(jxltran PRIVATE brotlienc brotlidec)
      message(STATUS "jxltran: brob support enabled")
    else()
      message(STATUS "jxltran: brotli targets not found, brob support disabled")
    endif()
  endif()
  set(TOOL_BINARIES ${TOOL_BINARIES} jxltran PARENT_SCOPE)
endif()

if(NOT JXLTRAN_STANDALONE AND NOT SANITIZER STREQUAL "none")
  set_target_properties(jxltran PROPERTIES LINKER_LANGUAGE CXX)
endif()

# Rebuild web-demo/jxltran.js when sources change (same .cc tree as jxltran;
# requires emcc on PATH). Incremental: only reruns if a dependency is newer
# than web-demo/jxltran.js.
if(JXLTRAN_STANDALONE)
  find_program(JXLTRAN_EMCC NAMES emcc)
  if(JXLTRAN_EMCC)
    set(JXLTRAN_WEB_JS "${CMAKE_CURRENT_SOURCE_DIR}/web-demo/jxltran.js")
    add_custom_command(
      OUTPUT "${JXLTRAN_WEB_JS}"
      COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/web-demo/build.sh"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/web-demo"
      DEPENDS ${JXLTRAN_SOURCES}
              "${CMAKE_CURRENT_SOURCE_DIR}/web-demo/build.sh"
      COMMENT "Emscripten: web-demo/jxltran.js")
    add_custom_target(jxltran_wasm DEPENDS "${JXLTRAN_WEB_JS}")
    add_dependencies(jxltran jxltran_wasm)
    message(STATUS "jxltran: wasm web-demo tracks sources (emcc: ${JXLTRAN_EMCC})")
  else()
    message(
      STATUS
        "jxltran: emcc not found; skip web-demo auto-build (run web-demo/build.sh manually)"
    )
  endif()
endif()
