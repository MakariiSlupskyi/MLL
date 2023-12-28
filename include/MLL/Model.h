#ifndef MLL_MODEL_H_
#define MLL_MODEL_H_

#include "MLL/Layer.h"
#include "MLL/Data.h"
#include <string>

namespace ml {
	class Model
	{
	public:
		Model(const std::vector<ml::Layer*>& layers = {});
		Model(const std::string& path);
		
		void compile(const std::string& optimizerType, const std::string& lossFuncType);

		std::vector<ml::Layer*> getLayers() const { return layers; }
		std::string getOptimizerType() const { return optimizerType; }
		std::string getLossFuncType() const { return lossFuncType; }

		void write(const std::string& path) const;
		Model read(const std::string& path);

		double evaluate(const ml::Data& inputData, const ml::Data& labels);
		ml::Tensor inference(const ml::Tensor& inputData);
		void train(const ml::Data& trainingData, const ml::Data& labels, int epoches);

		void applyRegularization(double lambda);

	private:
		void applyL2Regularization(double lambda);
		
		std::string optimizerType, lossFuncType;
		std::vector<ml::Layer*> layers;
	};
}

#endif