cmake_minimum_required(VERSION 3.10)
project(SP_Vocab_Trainer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(OpenCV 4.0 REQUIRED)

set(DBOW2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Thirdparty/DBoW2)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${DBOW2_DIR}
    ${DBOW2_DIR}/DUtils
    ${OpenCV_INCLUDE_DIRS}
)

# FSuperPoint sources
set(DBOW2_SP_SOURCES
    ${DBOW2_DIR}/DBoW2/FSuperPoint.cpp
)

# DBoW2 core sources (needed for TemplatedVocabulary)
# FClass is header-only, no .cpp needed
set(DBOW2_CORE_SOURCES
    ${DBOW2_DIR}/DBoW2/BowVector.cpp
    ${DBOW2_DIR}/DBoW2/FeatureVector.cpp
    ${DBOW2_DIR}/DBoW2/ScoringObject.cpp
    ${DBOW2_DIR}/DUtils/Random.cpp
    ${DBOW2_DIR}/DUtils/Timestamp.cpp
)

add_executable(train_sp_vocabulary
    train_sp_vocabulary.cc
    ${DBOW2_SP_SOURCES}
    ${DBOW2_CORE_SOURCES}
)

target_link_libraries(train_sp_vocabulary ${OpenCV_LIBS})
