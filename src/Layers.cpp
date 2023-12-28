#include "MLL/Layers.h"

namespace ml {
	const std::unordered_map<std::string, std::function<ml::Layer*(std::ifstream& file)>> Layers {
		{ "Activation", [](std::ifstream& file) -> ml::Layer* { return new ml::Activation(file); } },
		{ "Convolutional", [](std::ifstream& file) -> ml::Layer* { return new ml::Convolutional(file); } },
		{ "Dense", [](std::ifstream& file) -> ml::Layer* { return new ml::Dense(file); } },
		{ "Flatten", [](std::ifstream& file) -> ml::Layer* { return new ml::Flatten(file); } },
		{ "Pooling", [](std::ifstream& file) -> ml::Layer* { return new ml::Pooling(file); } },
	};
}