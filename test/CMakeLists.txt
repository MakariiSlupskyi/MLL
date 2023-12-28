if(BUILD_TESTING)
	add_executable(tests tests.cpp)
	target_link_libraries(tests PRIVATE GTest::gtest_main mll-linear-algebra)
	target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/include)


	include(GoogleTest)
	gtest_discover_tests(tests)
endif()