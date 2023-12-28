#include "MLL/Layers/Dense.h"
#include "MLL/Functions/Others.h"
#include <string>
#include <stdexcept>
#include <fstream>

ml::Dense::Dense(int inputShape, int outputShape)
	: input(inputShape), output(outputShape), biases(outputShape), weights(outputShape, inputShape)
{
	biases.setRandom();
	weights.setRandom();
}

ml::Dense::Dense(std::ifstream& file) {
	std::string line;
	
	std::getline(file, line);
	std::vector<int>data = ml::strToIntVec(line);
	input = input.reshape(data[0]);
	output = output.reshape(data[1]);
	biases = biases.reshape(data[1]);
	weights = weights.reshape(data[1], data[0]);

	std::getline(file, line);
	biases.setValues(ml::strToDoubleVec(line));
	
	std::getline(file, line);
	weights.setValues(ml::strToDoubleVec(line));
	
}

ml::Tensor ml::Dense::forward(const ml::Tensor& input) {
	this->input = input;
	output = weights * input;
	output += biases;
	return output;
}

ml::Tensor ml::Dense::backward(const ml::Tensor& outputGrad, double learningRate) {
	ml::Matrix weightsGrad = ml::Matrix(outputGrad) * input.transpose();
	weights -= (weightsGrad * learningRate);
	biases -= (outputGrad * learningRate);
	return weights.transpose() * outputGrad;
}

void ml::Dense::write(std::ofstream& file) const {
	file << "Dense" << std::endl;

	file << input.getShape()[0] << ' ' << output.getShape()[0] << std::endl;

	for (auto elem : biases.getData()) { file << elem << ' '; }
	file << std::endl;

	for (auto elem : weights.getData()) { file << elem << ' '; }
	file << std::endl;
}