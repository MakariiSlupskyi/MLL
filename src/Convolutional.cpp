#include "MLL/Layers/Convolutional.h"
#include "MLL/Functions/Others.h"

ml::Convolutional::Convolutional(const std::vector<int>& inputShape, const std::vector<int>& kernelShape, int depth)
	: inputShape(inputShape), kernelShape(kernelShape), depth(depth), inputDepth(inputShape[0])
{
	std::vector<int> outputShape = {
		depth,
		inputShape[1] - kernelShape[0] + 1,
		inputShape[2] - kernelShape[1] + 1,
	};
	input = input.reshape(inputShape);

	output = output.reshape(outputShape);
	biases = biases.reshape(outputShape);
	kernels = kernels.reshape({
		depth,
		inputDepth,
		kernelShape[0],
		kernelShape[1],
	});

	kernels.setRandom();
	biases.setRandom();
}

ml::Convolutional::Convolutional(std::ifstream& file) {
	std::string line;

	for (int i = 0; i <= 2; ++i) {
		std::getline(file, line);

		if (i == 0) {
			inputShape = ml::strToIntVec(line);
			std::getline(file, line);
			kernelShape = ml::strToIntVec(line);
			std::getline(file, line);
			depth = std::stoi(line);

			*this = ml::Convolutional(inputShape, kernelShape, depth);
		} else if (i == 1) {
			biases.setValues(ml::strToDoubleVec(line));
		} else if (i == 2) {
			kernels.setValues(ml::strToDoubleVec(line));
		}
	}
}

ml::Tensor ml::Convolutional::forward(const ml::Tensor& input) {
	output.setValues(biases.getData());

	for (int i = 0; i < depth; ++i) {
		ml::Tensor slice = output.slice({i}), kernel = kernels.slice({i});
		for (int j = 0; j < inputDepth; ++j) {
			slice += ml::correlate2d(input.slice({j}), kernel.slice({j}), "valid");
		}

		output.setSlice({i}, slice);
	}

	return output;
}

ml::Tensor ml::Convolutional::backward(const ml::Tensor& outputGrad, double learningRate) {
	ml::Tensor kernelsGrad(kernels.getShape());
	ml::Tensor inputGrad(input.getShape());

	for (int i = 0; i < depth; ++i) {
		for (int j = 0; j < inputDepth; ++j) {
			kernelsGrad.setSlice({i, j}, ml::correlate2d(input.slice({j}), outputGrad.slice({i}), "valid"));
			inputGrad.setSlice({j}, inputGrad.slice({j}) + ml::convolve2d(outputGrad.slice({i}), kernels.slice({i, j}), "full"));
		}
	}
	kernels -= kernelsGrad * learningRate;
	biases -= outputGrad * learningRate;

	return inputGrad;
}

void ml::Convolutional::write(std::ofstream& file) const {
	file << "Convolutional" << std::endl;

	for (auto elem : inputShape) { file << elem << ' '; }
	file << std::endl;

	for (auto elem : kernelShape) { file << elem << ' '; }
	file << std::endl;

	file << depth << std::endl;

	for (auto elem : biases.getData()) { file << elem << ' '; }
	file << std::endl;

	for (auto elem : kernels.getData()) { file << elem << ' '; }
	file << std::endl;
}