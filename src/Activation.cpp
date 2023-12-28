#include "MLL/Layers/Activation.h"
#include "MLL/Functions/ActivFunctions.h"
#include <algorithm>
#include <cctype>

ml::Activation::Activation(const std::string& type) : type(type)
{
	std::transform(this->type.begin(), this->type.end(), this->type.begin(), [](unsigned char c) { return std::tolower(c); });
}

ml::Activation::Activation(std::ifstream& file) {
	std::string type;
	std::getline(file, type);

	*this = ml::Activation(type);
}

ml::Tensor ml::Activation::forward(const ml::Tensor& inputs) {
	this->inputs = inputs;
	return ml::ActivFunctions.at(type)(this->inputs);
}

ml::Tensor ml::Activation::backward(const ml::Tensor& outputGrad, double learningRate) {
	return (outputGrad * ml::ActivFuncDerivs.at(type)(inputs));
}

void ml::Activation::write(std::ofstream& file) const {
	file << "Activation" << std::endl;
	file << type << std::endl;
}