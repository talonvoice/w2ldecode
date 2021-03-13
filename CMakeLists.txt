cmake_minimum_required(VERSION 3.5.1)
project(libw2ldecode)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCE src/fl-derived/*.cpp src/w2l_decode.cpp)
file(GLOB KENLM kenlm/lm/*.cc kenlm/util/*.cc kenlm/util/double-conversion/*.cc)
list(FILTER KENLM EXCLUDE REGEX "(test|main).cc$")

add_library(w2ldecode SHARED ${SOURCE} ${KENLM})
add_definitions(-DKENLM_MAX_ORDER=5 -flto -O3)
target_include_directories(w2ldecode PRIVATE src kenlm)

install(FILES src/w2l_common.h src/w2l_decode.h DESTINATION include)
install(TARGETS w2ldecode DESTINATION lib)
