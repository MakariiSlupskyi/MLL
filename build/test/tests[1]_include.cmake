if(EXISTS "D:/dev/MLL/build/test/tests[1]_tests.cmake")
  include("D:/dev/MLL/build/test/tests[1]_tests.cmake")
else()
  add_test(tests_NOT_BUILT tests_NOT_BUILT)
endif()
