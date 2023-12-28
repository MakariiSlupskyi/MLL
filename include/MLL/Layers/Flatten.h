#ifndef MLL_FLATTEN_H_
#define MLL_FLATTEN_H_

#include "MLL/Layer.h"

#include <vector>

namespace ml {
	class Flatten : public ml::Layer
	{
	public:
		Flatten();
		Flatten(std::ifstream& file);
		
		ml::Tensor forward(const ml::Tensor& input) override;
		ml::Tensor backward(const ml::Tensor& outputGrad, double learningRate) override;
		ml::Tensor getParameters() override { return ml::Tensor({0}); };
		void setParameters(const ml::Tensor& other) {};

		void write(std::ofstream& file) const override;

	private:
		std::vector<int> inputShape;
	};
}

#endif