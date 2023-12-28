#include "MLL/Optimizers.h"
#include "MLL/Optimizers/SGD.h"

namespace ml {
	const std::unordered_map<std::string, std::function<ml::Optimizer*(ml::Model*, std::vector<ml::Layer*>*)>> Optimizers {
		{"sgd", [](ml::Model *model, std::vector<ml::Layer*>* layers) -> ml::Optimizer* { return new ml::SGD(model, layers); }}
	};
}