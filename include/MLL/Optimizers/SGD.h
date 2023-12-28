#ifndef MLL_SGD_H_
#define MLL_SGD_H_

#include "MLL/Optimizer.h"
#include "MLL/Functions/LossFunctions.h"
#include <iostream>
#include <cmath>

namespace ml {
	class SGD : public ml::Optimizer
	{
	public:
		SGD(ml::Model* model, std::vector<ml::Layer*>* layers) : Optimizer(model, layers)
		{}

		void train(const ml::Data& trainingData, const ml::Data& labels) override {
			for (int i = 0; i < trainingData.size(); ++i) {
				ml::Tensor output = model->inference(trainingData[i]);
				ml::Tensor error = ml::LossFuncDerivs.at(model->getLossFuncType())(output, labels[i]);
			
				for (int j = int(layers->size()) - 1; j >= 0; --j) {
					error = layers->operator[](j)->backward(error, 0.0001);
				}
			}
		}
	};
}

#endif