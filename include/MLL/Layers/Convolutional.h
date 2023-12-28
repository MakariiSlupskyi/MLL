#ifndef MLL_CONVOLUTIONAL_H_
#define MLL_CONVOLUTIONAL_H_

#include "MLL/Layer.h"
#include "MLL/Linear/Tensor.h"

#include <vector>

namespace ml {
	class Convolutional : public Layer
	{
	public:
		Convolutional(const std::vector<int>& inputShape, const std::vector<int>& kernelShape, int depth);
		Convolutional(std::ifstream& file);

		ml::Tensor forward(const ml::Tensor& input) override;
		ml::Tensor backward(const ml::Tensor& outputGrad, double learningRate) override;
		ml::Tensor getParameters() override { return kernels; };
		void setParameters(const ml::Tensor& other) { kernels = other; };

		void write(std::ofstream& file) const override;

	private:
		std::vector<int> inputShape, kernelShape;
		int depth, inputDepth;
		ml::Tensor input, output, biases, kernels;
	};
}

#endif