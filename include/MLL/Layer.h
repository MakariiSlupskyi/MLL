#ifndef MLL_LAYER_H_
#define MLL_LAYER_H_

#include "MLL/Linear/Tensor.h"
#include <fstream>
#include <vector>

namespace ml {
	class Layer
	{
	public:
		virtual ml::Tensor forward(const ml::Tensor& inputs) = 0;
		virtual ml::Tensor backward(const ml::Tensor& outputGrad, double learningRate) = 0;
		virtual ml::Tensor getParameters() = 0;
		virtual void setParameters(const ml::Tensor& other) = 0;

		virtual void write(std::ofstream& file) const = 0;
	};
}

#endif