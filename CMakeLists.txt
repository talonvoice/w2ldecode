cmake_minimum_required(VERSION 3.5.1)
project(libw2ldecode)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCE src/fl-derived/*.cpp src/w2l_decode.cpp)
file(GLOB KENLM kenlm/lm/*.cc kenlm/util/*.cc kenlm/util/double-conversion/*.cc)
list(FILTER KENLM EXCLUDE REGEX "(test|main).cc$")
add_definitions(-DKENLM_MAX_ORDER=6 -O3)

add_library(w2ldecode SHARED ${SOURCE} ${KENLM})
target_include_directories(w2ldecode PRIVATE src kenlm)

add_library(w2ldecode_static STATIC ${SOURCE} ${KENLM})
target_include_directories(w2ldecode_static PRIVATE src kenlm)
set_target_properties(w2ldecode_static PROPERTIES OUTPUT_NAME w2ldecode)

install(FILES src/w2l_common.h src/w2l_decode.h DESTINATION include)
install(TARGETS w2ldecode w2ldecode_static DESTINATION lib)
