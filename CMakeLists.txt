cmake_minimum_required(VERSION 3.16)
project(MLL VERSION 0.0.1 LANGUAGES CXX)

add_subdirectory(external)
add_subdirectory(test)
include(CTest)

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/res)

# add linear algebra library
add_library(mll-linear-algebra STATIC
	include/MLL/Linear/Tensor.h
	include/MLL/Linear/Matrix.h
	include/MLL/Linear/Vector.h
	src/Tensor.cpp
	src/Matrix.cpp
	src/Vector.cpp
)

set_target_properties(mll-linear-algebra PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)

# add machine learning library
add_library(mll-machine-learning STATIC
	include/MLL/Data.h
	include/MLL/Functions.h
	include/MLL/Layer.h
	include/MLL/Layers.h
	include/MLL/Model.h
	include/MLL/Optimizer.h
	include/MLL/Optimizers.h

	include/MLL/Functions/ActivFunctions.h
	include/MLL/Functions/LossFunctions.h
	include/MLL/Functions/Others.h

	include/MLL/Layers/Activation.h
	include/MLL/Layers/Convolutional.h
	include/MLL/Layers/Dense.h
	include/MLL/Layers/Flatten.h
	include/MLL/Layers/Pooling.h

	include/MLL/Optimizers/SGD.h

	src/Activation.cpp
	src/Convolutional.cpp
	src/Data.cpp
	src/Dense.cpp
	src/Flatten.cpp
	src/Functions.cpp
	src/Layers.cpp
	src/Model.cpp
	src/Optimizers.cpp
	src/Pooling.cpp
)

target_link_libraries(mll-machine-learning PUBLIC mll-linear-algebra)

set_target_properties(mll-machine-learning PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)