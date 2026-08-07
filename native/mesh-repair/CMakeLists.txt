cmake_minimum_required(VERSION 3.25)

project(scanlan_mesh_repair VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

find_package(CGAL 6 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
include(CGAL_Eigen3_support)

add_executable(scanlan-mesh-repair
  src/main.cpp
)

target_compile_features(scanlan-mesh-repair PRIVATE cxx_std_20)
target_link_libraries(scanlan-mesh-repair PRIVATE
  CGAL::CGAL
  CGAL::Eigen3_support
  nlohmann_json::nlohmann_json
)

if(MSVC)
  target_compile_options(scanlan-mesh-repair PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(scanlan-mesh-repair PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS scanlan-mesh-repair RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
