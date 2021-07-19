cmake_minimum_required(VERSION 3.5.1)
project(libw2ldecode)

file(GLOB_RECURSE SOURCE src/fl-derived/*.cpp src/w2l_decode.cpp)
file(GLOB KENLM kenlm/lm/*.cc kenlm/util/*.cc kenlm/util/double-conversion/*.cc)
list(FILTER KENLM EXCLUDE REGEX "(test|main).cc$")
add_definitions(-DKENLM_MAX_ORDER=6 -O3)
add_library(kenlm-obj OBJECT ${KENLM})
target_include_directories(kenlm-obj PRIVATE kenlm)

target_compile_features(kenlm-obj PRIVATE cxx_std_14)
set_property(TARGET kenlm-obj PROPERTY POSITION_INDEPENDENT_CODE 1)

add_library(w2ldecode-obj OBJECT ${SOURCE})
target_include_directories(w2ldecode-obj PRIVATE src kenlm)
set_property(TARGET w2ldecode-obj PROPERTY POSITION_INDEPENDENT_CODE 1)
target_compile_features(w2ldecode-obj PRIVATE cxx_std_17)

add_library(w2ldecode        SHARED $<TARGET_OBJECTS:w2ldecode-obj> $<TARGET_OBJECTS:kenlm-obj>)
add_library(w2ldecode-static STATIC $<TARGET_OBJECTS:w2ldecode-obj> $<TARGET_OBJECTS:kenlm-obj>)
set_target_properties(w2ldecode-static PROPERTIES OUTPUT_NAME w2ldecode)

install(FILES src/w2l_common.h src/w2l_decode.h DESTINATION include)
install(TARGETS w2ldecode w2ldecode-static DESTINATION lib)
