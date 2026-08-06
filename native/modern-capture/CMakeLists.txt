cmake_minimum_required(VERSION 3.24)
project(rgbd_capture_worker LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "The desktop RGB-D capture worker is currently Windows-only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rgbd-capture-worker src/main.cpp)
target_include_directories(rgbd-capture-worker PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")
target_link_libraries(rgbd-capture-worker PRIVATE windowscodecs ole32)
target_compile_definitions(rgbd-capture-worker PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(rgbd-capture-worker PRIVATE /W4 /permissive- /EHsc)

set(AZURE_KINECT_SDK_ROOT "$ENV{AZURE_KINECT_SDK_ROOT}" CACHE PATH "Azure Kinect Sensor SDK root")
if(NOT AZURE_KINECT_SDK_ROOT)
  foreach(candidate
      "C:/Program Files/Azure Kinect SDK v1.4.2"
      "C:/Program Files/Azure Kinect SDK v1.4.1"
      "C:/Program Files/Azure Kinect SDK v1.4.0")
    if(EXISTS "${candidate}/sdk/include/k4a/k4a.h")
      set(AZURE_KINECT_SDK_ROOT "${candidate}")
      break()
    endif()
  endforeach()
endif()

if(EXISTS "${AZURE_KINECT_SDK_ROOT}/sdk/include/k4a/k4a.h")
  set(K4A_ARCH_ROOT "${AZURE_KINECT_SDK_ROOT}/sdk/windows-desktop/amd64/release")
  target_compile_definitions(rgbd-capture-worker PRIVATE SCANLAN_HAS_AZURE_KINECT=1)
  target_include_directories(rgbd-capture-worker PRIVATE "${AZURE_KINECT_SDK_ROOT}/sdk/include")
  target_link_libraries(rgbd-capture-worker PRIVATE "${K4A_ARCH_ROOT}/lib/k4a.lib")
  foreach(runtime k4a.dll depthengine_2_0.dll)
    add_custom_command(TARGET rgbd-capture-worker POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${K4A_ARCH_ROOT}/bin/${runtime}"
        "$<TARGET_FILE_DIR:rgbd-capture-worker>/${runtime}")
  endforeach()
  message(STATUS "Azure Kinect backend enabled: ${AZURE_KINECT_SDK_ROOT}")
else()
  message(WARNING "Azure Kinect SDK not found; the worker will report that backend as unavailable")
endif()

set(ORBBEC_SDK_ROOT "$ENV{ORBBEC_SDK_ROOT}" CACHE PATH "Installed Orbbec SDK v2 root")
if(NOT ORBBEC_SDK_ROOT)
  file(GLOB ORBBEC_INSTALLED_ROOTS LIST_DIRECTORIES true "C:/Program Files/OrbbecSDK*")
  list(SORT ORBBEC_INSTALLED_ROOTS COMPARE NATURAL ORDER DESCENDING)
  foreach(candidate IN LISTS ORBBEC_INSTALLED_ROOTS)
    if(EXISTS "${candidate}/lib/OrbbecSDKConfig.cmake")
      set(ORBBEC_SDK_ROOT "${candidate}")
      break()
    endif()
  endforeach()
endif()
if(ORBBEC_SDK_ROOT)
  set(OrbbecSDK_DIR "${ORBBEC_SDK_ROOT}/lib")
endif()
find_package(OrbbecSDK CONFIG QUIET)
if(TARGET ob::OrbbecSDK)
  target_compile_definitions(rgbd-capture-worker PRIVATE SCANLAN_HAS_ORBBEC=1)
  target_link_libraries(rgbd-capture-worker PRIVATE ob::OrbbecSDK)
  if(EXISTS "${ORBBEC_SDK_ROOT}/bin")
    # copy_directory attempts to replace every destination file even when it
    # is unchanged. On Windows that fails whenever a running preview has the
    # Orbbec DLL loaded. Copy each runtime file conditionally so an unchanged
    # SDK can coexist with an active debug camera session; a genuinely changed
    # DLL still fails safely and asks for the preview to be restarted.
    file(GLOB_RECURSE ORBBEC_RUNTIME_FILES
      CONFIGURE_DEPENDS
      RELATIVE "${ORBBEC_SDK_ROOT}/bin"
      "${ORBBEC_SDK_ROOT}/bin/*")
    foreach(runtime IN LISTS ORBBEC_RUNTIME_FILES)
      get_filename_component(runtime_directory "${runtime}" DIRECTORY)
      if(runtime_directory)
        add_custom_command(TARGET rgbd-capture-worker POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:rgbd-capture-worker>/${runtime_directory}")
      endif()
      add_custom_command(TARGET rgbd-capture-worker POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${ORBBEC_SDK_ROOT}/bin/${runtime}"
          "$<TARGET_FILE_DIR:rgbd-capture-worker>/${runtime}")
    endforeach()
  endif()
  message(STATUS "Orbbec Femto Mega backend enabled: ${ORBBEC_SDK_ROOT}")
else()
  message(WARNING "Orbbec SDK v2 not found; set ORBBEC_SDK_ROOT to enable Femto Mega USB/network capture")
endif()

install(TARGETS rgbd-capture-worker RUNTIME DESTINATION .)
