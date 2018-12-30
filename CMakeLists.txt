#---------------------------------------------------
# OpenMM Charmm Plugin
#
# Creates OpenMMCharmm plugin library.
#
# Windows:
#   OpenMMCharmm.dll
#   OpenMMCharmm.lib
#   OpenMMCharmm_static.lib
# Unix:
#   libOpenMMCharmm.so
#   libOpenMMCharmm_static.a
#----------------------------------------------------

#INCLUDE(Dart)

#include_directories($ENV{INCLUDE})
#include_directories($ENV{INCLUDE}/openmm/)
#include_directories($ENV{INCLUDE}/openmm/reference)
include_directories("/home/ymwang/apps/openmm_src/openmm_bin/include/")
include_directories("/home/ymwang/apps/openmm_src/openmm_bin/include/openmm")
include_directories("/home/ymwang/apps/openmm_src/openmm_bin/include/openmm/reference")
SET(CMAKE_CXX_COMPILER icpc) 
SET(CMAKE_CC_COMPILER icc)
add_compile_options(-std=c++0x)
add_compile_options(-O3)
SET(CMAKE_INSTALL_PREFIX "/home/ymwang/apps/openmm_src/openmm_bin/")

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(OPENMM_CHARMM_PLUGIN_SOURCE_SUBDIRS . openmmapi)

# Collect up information about the version of the OpenMM library we're building
# and make it available to the code so it can be built into the binaries.

SET(OPENMM_CHARMM_LIBRARY_NAME OpenMMCharmm)
SET(OPENMM_CHARMM_MAJOR_VERSION 1)
SET(OPENMM_CHARMM_MINOR_VERSION 0)
SET(OPENMM_CHARMM_BUILD_VERSION 0)

ADD_DEFINITIONS(-DOPENMM_CHARMM_LIBRARY_NAME=${OPENMM_CHARMM_LIBRARY_NAME}
                -DOPENMM_CHARMM_MAJOR_VERSION=${OPENMM_CHARMM_MAJOR_VERSION}
                -DOPENMM_CHARMM_MINOR_VERSION=${OPENMM_CHARMM_MINOR_VERSION}
		          -DOPENMM_CHARMM_BUILD_VERSION=${OPENMM_CHARMM_BUILD_VERSION})

SET(SHARED_CHARMM_TARGET ${OPENMM_CHARMM_LIBRARY_NAME})
SET(STATIC_CHARMM_TARGET ${OPENMM_CHARMM_LIBRARY_NAME}_static)

# These are all the places to search for header files which are
# to be part of the API.
SET(API_CHARMM_INCLUDE_DIRS) # start empty
FOREACH(subdir ${OPENMM_CHARMM_PLUGIN_SOURCE_SUBDIRS})
    # append
    SET(API_CHARMM_INCLUDE_DIRS ${API_CHARMM_INCLUDE_DIRS}
                                     ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include
                                     ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include/internal)
ENDFOREACH(subdir)

# We'll need both *relative* path names, starting with their API_INCLUDE_DIRS,
# and absolute pathnames.
SET(API_CHARMM_REL_INCLUDE_FILES)   # start these out empty
SET(API_CHARMM_ABS_INCLUDE_FILES)

FOREACH(dir ${API_CHARMM_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)	# returns full pathnames
    SET(API_CHARMM_ABS_INCLUDE_FILES ${API_CHARMM_ABS_INCLUDE_FILES} ${fullpaths})

    FOREACH(pathname ${fullpaths})
        GET_FILENAME_COMPONENT(filename ${pathname} NAME)
        SET(API_CHARMM_REL_INCLUDE_FILES ${API_CHARMM_REL_INCLUDE_FILES} ${dir}/${filename})
    ENDFOREACH(pathname)
ENDFOREACH(dir)

# collect up source files
SET(SOURCE_CHARMM_FILES) # empty
SET(SOURCE_CHARMM_INCLUDE_FILES)

FOREACH(subdir ${OPENMM_CHARMM_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_CHARMM_FILES         ${SOURCE_CHARMM_FILES}         ${src_files})   #append
    SET(SOURCE_CHARMM_INCLUDE_FILES ${SOURCE_CHARMM_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# ----------------------------------------------------------------------------

# If API_CHARMM wrappers are being generated, and add them to the build.
IF(OPENMM_BUILD_C_AND_FORTRAN_WRAPPERS)
    #ADD_SUBDIRECTORY(wrappers)
    #SET(SOURCE_CHARMM_FILES ${SOURCE_CHARMM_FILES} wrappers/OpenMMCWrapper.cpp  wrappers/CharmmOpenMMFortranWrapper.cpp)
    #SET_SOURCE_FILES_PROPERTIES(wrappers/OpenMMCWrapper.cpp wrappers/CharmmOpenMMFortranWrapper.cpp PROPERTIES GENERATED TRUE)
ENDIF(OPENMM_BUILD_C_AND_FORTRAN_WRAPPERS)

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)

ADD_LIBRARY(${SHARED_CHARMM_TARGET} SHARED ${SOURCE_CHARMM_FILES} ${SOURCE_CHARMM_INCLUDE_FILES} ${API_CHARMM_ABS_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_CHARMM_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_LINK_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_CHARMM_BUILDING_SHARED_LIBRARY -DLEPTON_BUILDING_SHARED_LIBRARY")

FILE(GLOB serialization_files  ${CMAKE_CUURENT_SOURCE_DIR}/serialization/src/*.cpp)
SET_SOURCE_FILES_PROPERTIES(${serialization_files} PROPERTIES COMPILE_FLAGS "-DOPENMM_CHARMM_BUILDING_SHARED_LIBRARY -DTIXML_USE_STL")

IF(OPENMM_BUILD_STATIC_LIB)
  ADD_LIBRARY(${STATIC_CHARMM_TARGET} STATIC ${SOURCE_CHARMM_FILES} ${SOURCE_CHARMM_INCLUDE_FILES} ${API_CHARMM_ABS_INCLUDE_FILES})
  SET_TARGET_PROPERTIES(${STATIC_CHARMM_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_LINK_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_CHARMM_BUILDING_STATIC_LIBRARY -DOPENMM_USE_STATIC_LIBRARIES -DOPENMM_BUILDING_STATIC_LIBRARY -DLEPTON_USE_STATIC_LIBRARIES -DLEPTON_BUILDING_STATIC_LIBRARY")
ENDIF(OPENMM_BUILD_STATIC_LIB)

IF(OPENMM_BUILD_C_AND_FORTRAN_WRAPPERS)
    ADD_DEPENDENCIES(${SHARED_CHARMM_TARGET} ApiWrappers)
    IF(OPENMM_BUILD_STATIC_LIB)
        ADD_DEPENDENCIES(${STATIC_CHARMM_TARGET} ApiWrappers)
    ENDIF(OPENMM_BUILD_STATIC_LIB)
ENDIF(OPENMM_BUILD_C_AND_FORTRAN_WRAPPERS)

# ----------------------------------------------------------------------------

# On Linux need to link to libdl
FIND_LIBRARY(DL_LIBRARY dl)
IF(DL_LIBRARY)
  TARGET_LINK_LIBRARIES(${SHARED_CHARMM_TARGET} ${DL_LIBRARY})
  IF(OPENMM_BUILD_STATIC_LIB)
    TARGET_LINK_LIBRARIES(${STATIC_CHARMM_TARGET} ${DL_LIBRARY})
  ENDIF(OPENMM_BUILD_STATIC_LIB)
ENDIF(DL_LIBRARY)

SET( OpenMMLib OpenMM )

TARGET_LINK_LIBRARIES( ${SHARED_CHARMM_TARGET} ${SHARED_TARGET} )
IF(OPENMM_BUILD_STATIC_LIB)
    TARGET_LINK_LIBRARIES( ${STATIC_CHARMM_TARGET} ${STATIC_TARGET} )
ENDIF(OPENMM_BUILD_STATIC_LIB)

# Which hardware platforms to build

ADD_SUBDIRECTORY(platforms/reference)

IF(OPENMM_BUILD_CUDA_LIB)
    #SET(OPENMM_BUILD_CHARMM_CUDA_LIB ON CACHE BOOL "Build OpenMMCuda library for Nvidia GPUs")
ELSE(OPENMM_BUILD_CUDA_LIB)
    #SET(OPENMM_BUILD_CHARMM_CUDA_LIB OFF CACHE BOOL "Build OpenMMCuda library for Nvidia GPUs")
ENDIF(OPENMM_BUILD_CUDA_LIB)

SET(OPENMM_BUILD_CHARMM_PATH)
SET(OPENMM_BUILD_CHARMM_CUDA_PATH)
IF(OPENMM_BUILD_CHARMM_CUDA_LIB)
    #ADD_SUBDIRECTORY(platforms/cuda)
    #SET(OPENMM_BUILD_CHARMM_CUDA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/platforms/cuda)
    #SET(OPENMM_CHARMM_CUDA_SOURCE_SUBDIRS . openmmapi olla platforms/cuda)
ENDIF(OPENMM_BUILD_CHARMM_CUDA_LIB)

INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_CHARMM_TARGET})
IF(OPENMM_BUILD_STATIC_LIB)
  INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${STATIC_CHARMM_TARGET})
ENDIF(OPENMM_BUILD_STATIC_LIB)
FILE(GLOB CORE_HEADERS     include/*.h          */include/*.h)
FILE(GLOB TOP_HEADERS      include/openmm/*.h          */include/openmm/*.h)
FILE(GLOB INTERNAL_HEADERS include/openmm/internal/*.h */include/openmm/internal/*.h )
INSTALL_FILES(/include                 FILES ${CORE_HEADERS})
INSTALL_FILES(/include/openmm          FILES ${TOP_HEADERS})
INSTALL_FILES(/include/openmm/internal FILES ${INTERNAL_HEADERS})

#
# Testing
#

#ENABLE_TESTING()

IF (EXECUTABLE_OUTPUT_PATH)
    #SET (TEST_PATH ${EXECUTABLE_OUTPUT_PATH})
ELSE (EXECUTABLE_OUTPUT_PATH)
    #SET (TEST_PATH .)
ENDIF (EXECUTABLE_OUTPUT_PATH)

IF(BUILD_TESTING AND OPENMM_BUILD_SERIALIZATION_TESTS)
    #ADD_SUBDIRECTORY(serialization/tests)
ENDIF(BUILD_TESTING AND OPENMM_BUILD_SERIALIZATION_TESTS)
