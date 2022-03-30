cmake_minimum_required(VERSION 3.15)
project(libw2ldecode)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")

option(USE_KENLM "Build with kenlm" ON)
option(MSVC_STATIC "Statically link MSVC runtimes" OFF)

if (MSVC_STATIC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if (USE_KENLM)
    file(GLOB KENLM kenlm/lm/*.cc kenlm/util/*.cc kenlm/util/double-conversion/*.cc)
    list(FILTER KENLM EXCLUDE REGEX "(test|main).cc$")
    add_definitions(-DKENLM_MAX_ORDER=6 -O3)
    add_library(kenlm-obj OBJECT ${KENLM})
    target_include_directories(kenlm-obj PRIVATE kenlm)

    target_compile_features(kenlm-obj PRIVATE cxx_std_14)
    set_property(TARGET kenlm-obj PROPERTY POSITION_INDEPENDENT_CODE 1)
endif()

file(GLOB_RECURSE SOURCE src/fl-derived/*.cpp src/w2l_decode.cpp)
if (NOT USE_KENLM)
    list(FILTER SOURCE EXCLUDE REGEX "KenLM")
endif()

add_library(w2ldecode-obj OBJECT ${SOURCE})
target_include_directories(w2ldecode-obj PRIVATE src kenlm)
set_property(TARGET w2ldecode-obj PROPERTY POSITION_INDEPENDENT_CODE 1)
target_compile_features(w2ldecode-obj PRIVATE cxx_std_17)

set(OBJECTS $<TARGET_OBJECTS:w2ldecode-obj>)
if (USE_KENLM)
    set(OBJECTS ${OBJECTS} $<TARGET_OBJECTS:kenlm-obj>)
    add_definitions("-DUSE_KENLM=1")
endif()
add_library(w2ldecode        SHARED ${OBJECTS})
add_library(w2ldecode-static STATIC ${OBJECTS})

install(FILES src/w2l_common.h src/w2l_decode.h DESTINATION include)
install(TARGETS w2ldecode w2ldecode-static DESTINATION lib)
