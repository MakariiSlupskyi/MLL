#ifndef MLL_LAYERS_H
#define MLL_LAYERS_H

#include "MLL/Layers/Activation.h"
#include "MLL/Layers/Convolutional.h"
#include "MLL/Layers/Dense.h"
#include "MLL/Layers/Flatten.h"
#include "MLL/Layers/Pooling.h"
#include <unordered_map>
#include <functional>
#include <string>

namespace ml {
	extern const std::unordered_map<std::string, std::function<ml::Layer*(std::ifstream& file)>> Layers;
}


#endif