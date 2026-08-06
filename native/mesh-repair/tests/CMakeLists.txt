find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME mesh-repair-analyze
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_analyze.py"
          "$<TARGET_FILE:scanlan-mesh-repair>" "${CMAKE_CURRENT_BINARY_DIR}/results"
)
