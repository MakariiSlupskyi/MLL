#ifndef MLL_LAYERS_DENSE_H_
#define MLL_LAYERS_DENSE_H_

#include "MLL/Layer.h"
#include "MLL/Linear/Tensor.h"
#include "MLL/Linear/Matrix.h"
#include "MLL/Linear/Vector.h"

namespace ml {
	class Dense : public ml::Layer
	{
	public:
		Dense(int inputsNum, int outputsNum);
		Dense(std::ifstream& file);

		ml::Tensor forward(const ml::Tensor& inputs) override;
		ml::Tensor backward(const ml::Tensor& output, double learningRate) override;
		ml::Tensor getParameters() override { return weights; };
		void setParameters(const ml::Tensor& other) { weights = other; };

		void write(std::ofstream& file) const override;

		private:
			ml::Vector input, output, biases;
			ml::Matrix weights;
	};
}

#endif