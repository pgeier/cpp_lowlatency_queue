include_directories("../include")

add_executable(TestBinaryUtils test_binary_utils.cpp)
target_link_libraries(TestBinaryUtils
    PRIVATE
    llq)

add_test(
  NAME test_binary_utils
  COMMAND $<TARGET_FILE:TestBinaryUtils>
  )


add_executable(TestLowLatencyQueue test_lowlatencyqueue.cpp)
target_link_libraries(TestLowLatencyQueue
    PRIVATE
    llq)

add_test(
  NAME test_lowlatencyqueue
  COMMAND $<TARGET_FILE:TestLowLatencyQueue>
  )
