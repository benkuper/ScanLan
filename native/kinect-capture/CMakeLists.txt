cmake_minimum_required(VERSION 3.24)
project(legacy_capture_worker LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "The Kinect v2 SDK capture worker is Windows-only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(KINECT_SDK_ROOT "$ENV{KINECTSDK20_DIR}" CACHE PATH "Kinect for Windows SDK v2 root")
if(NOT KINECT_SDK_ROOT)
  set(KINECT_SDK_ROOT "C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409")
endif()

if(EXISTS "${KINECT_SDK_ROOT}/inc/Kinect.h")
  add_executable(legacy-capture-worker src/main.cpp)
  target_include_directories(legacy-capture-worker PRIVATE "${KINECT_SDK_ROOT}/inc" "${CMAKE_CURRENT_SOURCE_DIR}/../common")
  target_link_directories(legacy-capture-worker PRIVATE "${KINECT_SDK_ROOT}/Lib/x64")
  target_link_libraries(legacy-capture-worker PRIVATE Kinect20 Kinect20.Fusion windowscodecs ole32)
  target_compile_definitions(legacy-capture-worker PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(legacy-capture-worker PRIVATE /W4 /permissive- /EHsc)

  set(KINECT_FUSION_RUNTIME "${KINECT_SDK_ROOT}/Redist/Fusion/x64/Kinect20.Fusion.dll")
  if(EXISTS "${KINECT_FUSION_RUNTIME}")
    add_custom_command(TARGET legacy-capture-worker POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${KINECT_FUSION_RUNTIME}"
        "$<TARGET_FILE_DIR:legacy-capture-worker>/Kinect20.Fusion.dll")
  endif()
  message(STATUS "Kinect v2 backend enabled: ${KINECT_SDK_ROOT}")
else()
  add_executable(legacy-capture-worker src/stub.cpp)
  target_compile_options(legacy-capture-worker PRIVATE /W4 /permissive- /EHsc)
  message(WARNING "Kinect for Windows SDK 2.0 not found; building an unavailable-backend stub")
endif()

install(TARGETS legacy-capture-worker RUNTIME DESTINATION .)
