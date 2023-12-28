#ifndef MLL_OPTIMIZER_H_
#define MLL_OPTIMIZER_H_

#include "MLL/Data.h"
#include "MLL/Model.h"
#include "MLL/Layer.h"
#include <vector>

namespace ml {
	class Optimizer
	{
	public:
		Optimizer(ml::Model* model, std::vector<ml::Layer*>* layers) : model(model), layers(layers)
		{}

		virtual void train(const ml::Data& trainingData, const ml::Data& labels) = 0;		

	protected:
		ml::Model* const model;
		std::vector<ml::Layer*>* const layers;
	};
}

#endif