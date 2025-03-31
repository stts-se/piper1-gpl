# Builds Python module for Piper using espeak-ng and cmake.
cmake_minimum_required(VERSION 3.18)
project(piper LANGUAGES C CXX)

include(UseSWIG)
include(ExternalProject)
find_package(SWIG REQUIRED)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Install location for espeak-ng
set(ESPEAKNG_BUILD_DIR ${CMAKE_BINARY_DIR}/espeak_ng)
set(ESPEAKNG_INSTALL_DIR ${CMAKE_BINARY_DIR}/espeak_ng-install)

if(WIN32)
    set(ESPEAKNG_STATIC_LIB ${ESPEAKNG_INSTALL_DIR}/lib/espeak-ng.lib)
    set(UCD_STATIC_LIB ${ESPEAKNG_BUILD_DIR}/src/espeak_ng_external-build/src/ucd-tools/Release/ucd.lib)
else()
    set(ESPEAKNG_STATIC_LIB ${ESPEAKNG_INSTALL_DIR}/lib/libespeak-ng.a)
    set(UCD_STATIC_LIB ${ESPEAKNG_BUILD_DIR}/src/espeak_ng_external-build/src/ucd-tools/libucd.a)
endif()

ExternalProject_Add(espeak_ng_external
    GIT_REPOSITORY https://github.com/espeak-ng/espeak-ng.git
    GIT_TAG 212928b394a96e8fd2096616bfd54e17845c48f6  # 2025-Mar-22
    PREFIX ${ESPEAKNG_BUILD_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${ESPEAKNG_INSTALL_DIR}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DUSE_ASYNC:BOOL=OFF
        -DUSE_MBROLA:BOOL=OFF
        -DUSE_LIBSONIC:BOOL=OFF
        -DUSE_LIBPCAUDIO:BOOL=OFF
        -DUSE_KLATT:BOOL=OFF
        -DUSE_SPEECHPLAYER:BOOL=OFF
        -DEXTRA_cmn:BOOL=ON
        -DEXTRA_ru:BOOL=ON
        "-DCMAKE_C_FLAGS=-D_FILE_OFFSET_BITS=64 -I${ESPEAKNG_BUILD_DIR}/src/espeak_ng_external/src/ucd-tools/src/include"
    BUILD_BYPRODUCTS
        ${ESPEAKNG_STATIC_LIB}
        ${UCD_STATIC_LIB}
    UPDATE_DISCONNECTED TRUE
)

include_directories(
    ${ESPEAKNG_INSTALL_DIR}/include
)

# Build the SWIG module
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_SOURCE_DIR}/piper)

swig_add_library(espeak_ng
    TYPE MODULE
    LANGUAGE python
    SOURCES espeak_ng.i
)

add_dependencies(${SWIG_MODULE_espeak_ng_REAL_NAME} espeak_ng_external)

# Include the espeak-ng headers and link the static lib
include_directories(${CMAKE_BINARY_DIR}/espeak_ng-install/include)
target_link_libraries(${SWIG_MODULE_espeak_ng_REAL_NAME}
    ${ESPEAKNG_STATIC_LIB}
    ${UCD_STATIC_LIB}
    Python::Module
)

set_target_properties(${SWIG_MODULE_espeak_ng_REAL_NAME} PROPERTIES
    OUTPUT_NAME "_espeak_ng"
)

install(TARGETS ${SWIG_MODULE_espeak_ng_REAL_NAME}
    LIBRARY DESTINATION piper
    RUNTIME DESTINATION piper
)

install(FILES ${CMAKE_SWIG_OUTDIR}/espeak_ng.py DESTINATION piper)

# Copy espeak-ng-data
set(DATA_SRC ${CMAKE_BINARY_DIR}/espeak_ng-install/share/espeak-ng-data)
set(DATA_DST ${CMAKE_CURRENT_SOURCE_DIR}/piper/espeak-ng-data)

add_custom_target(copy_espeak_ng_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${DATA_SRC} ${DATA_DST}
    COMMENT "Copying espeak-ng-data after espeak-ng external project builds"
)

add_dependencies(copy_espeak_ng_data espeak_ng_external)
