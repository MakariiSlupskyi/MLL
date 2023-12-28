#include "MLL/Layers/Flatten.h"
#include "MLL/Linear/Vector.h"

ml::Flatten::Flatten() {}
ml::Flatten::Flatten(std::ifstream& file) {}

ml::Tensor ml::Flatten::forward(const ml::Tensor& input) {
	inputShape = input.getShape();
	return ml::Vector(input.getData());
}

ml::Tensor ml::Flatten::backward(const ml::Tensor& outputGrad, double learningRate) {
	return outputGrad.reshape(inputShape);
}

void ml::Flatten::write(std::ofstream& file) const {
	file << "Flatten" << std::endl;
}