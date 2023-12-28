#ifndef MLL_POOLING_H_
#define MLL_POOLING_H_

#include "MLL/Layer.h"
#include "MLL/Linear/Tensor.h"
#include <vector>

namespace ml {
	class Pooling : public ml::Layer
	{
	public:
		Pooling(int poolSize, int strides = -1);
		Pooling(std::ifstream& file);
	
		ml::Tensor forward(const ml::Tensor& input) override;
		ml::Tensor backward(const ml::Tensor& outputGrad, double learningRate) override;
		ml::Tensor getParameters() override { return ml::Tensor({0}); };
		void setParameters(const ml::Tensor& other) {};

		void write(std::ofstream& file) const override;

	private:
		int poolSize, strides;
		ml::Tensor input, output;
	};
}

#endif