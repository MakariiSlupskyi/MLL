#ifndef MLL_ACTIVATION_H_
#define MLL_ACTIVATION_H_

#include "MLL/Layer.h"
#include <string>

namespace ml {
	class Activation : public ml::Layer
	{
	public:
		Activation(const std::string& type);
		Activation(std::ifstream& file);

		std::string getType() const { return type; }

		ml::Tensor forward(const ml::Tensor& inputs) override;
		ml::Tensor backward(const ml::Tensor& outputGrad, double learningRate) override;
		ml::Tensor getParameters() override { return ml::Tensor({0}); };
		void setParameters(const ml::Tensor& other) {};

		void write(std::ofstream& file) const override;

	private:
		std::string type;
		ml::Tensor inputs;
	};
}

#endif