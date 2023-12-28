#ifndef MLL_OPTIMIZERS_H_
#define MLL_OPTIMIZERS_H_

#include "MLL/Optimizer.h"
#include <unordered_map>
#include <functional>
#include <string>

namespace ml {
	extern const std::unordered_map<std::string, std::function<ml::Optimizer*(ml::Model*, std::vector<ml::Layer*>*)>> Optimizers;
}

#endif