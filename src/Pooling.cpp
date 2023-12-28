#include "MLL/Layers/Pooling.h"
#include "MLL/Functions/Others.h"
#include <vector>
#include <string>

ml::Pooling::Pooling(int poolSize, int strides) : poolSize(poolSize), strides(strides == -1 ? poolSize : strides)
{}

ml::Pooling::Pooling(std::ifstream& file) {
	std::string line;
	std::getline(file, line);
	std::vector<int> data = ml::strToIntVec(line);
	*this = ml::Pooling(data[0], data[1]);
}

ml::Tensor ml::Pooling::forward(const ml::Tensor& input) {
	this->input = input;
	
	output = output.reshape({input.getShape()[0], input.getShape()[1] / strides, input.getShape()[2] / strides});

	for (int i = 0; i < output.getShape()[0]; ++i) {
		for (int j = 0; j < output.getShape()[1]; ++j) {
			for (int k = 0; k < output.getShape()[2]; ++k) {
				output({i, j, k}) = input.block(
					{i, j * strides, k * strides},
					{1, poolSize, poolSize}
				).max();
			}
		}
	}
	return output;
}

ml::Tensor ml::Pooling::backward(const ml::Tensor& outputGrad, double learningRate) {
	ml::Tensor inputGrad(input.getShape());
	for (int i = 0; i < output.getShape()[0]; ++i) {
		for (int j = 0; j < output.getShape()[1]; ++j) {
			for (int k = 0; k < output.getShape()[2]; ++k) {
				inputGrad({i, j * strides, k * strides}) = outputGrad({i, j, k});
			}
		}
	}
	return inputGrad;
}

void ml::Pooling::write(std::ofstream& file) const {
	file << "Pooling" << std::endl;
	file << poolSize << ' ' << strides << std::endl;
}