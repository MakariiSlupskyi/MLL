#include "MLL/Model.h"
#include "MLL/Layers.h"
#include "MLL/Optimizers.h"
#include "MLL/Functions/LossFunctions.h"
#include <algorithm>
#include <stdexcept>
#include <cstdlib>

ml::Model::Model(const std::vector<ml::Layer*>& layers) : layers(layers)
{}

ml::Model::Model(const std::string& path) {
	this->read(path);
}

void ml::Model::write(const std::string& path) const {
	std::ofstream file(path);

	if (!file.is_open()) {
		throw std::invalid_argument("Failed to open file " + path + "\n");
	} else {
		file << layers.size() << std::endl;
		
		for (int i = 0; i < layers.size(); ++i) {
			layers[i]->write(file);
		}
	}
}

ml::Model ml::Model::read(const std::string& path) {
	std::ifstream file(path);

	if (!file.is_open()) {
		throw std::invalid_argument("Can't open file: " + path);
	}
	std::string line;

	// Read the number of layers
	std::getline(file, line);
	int numLayers = std::stoi(line);

	// Read and create layers
	for (int i = 0; i < numLayers; ++i) {
		std::getline(file, line);
		layers.push_back(ml::Layers.at(line)(file));
	}

	file.close();

	return *this;
}

void ml::Model::compile(const std::string& optimizerType, const std::string& lossFuncType) {
	this->optimizerType = optimizerType;
	this->lossFuncType = lossFuncType;

	std::transform(optimizerType.begin(), optimizerType.end(), this->optimizerType.begin(), [](unsigned char c) {
		return std::tolower(c);
	});
	std::transform(lossFuncType.begin(), lossFuncType.end(), this->lossFuncType.begin(), [](unsigned char c) {
		return std::tolower(c);
	});
}

double ml::Model::evaluate(const ml::Data& inputData, const ml::Data& labels) {
	double res = 0.0f;
	for (int i = 0; i < inputData.size(); ++i) {
		res += ml::LossFunctions.at(lossFuncType)(this->inference(inputData[i]), labels[i]);
	}
	return res / inputData.size();
}

ml::Tensor ml::Model::inference(const ml::Tensor& inputData) {
	ml::Tensor output = inputData;
	for (int i = 0; i < layers.size(); ++i) {
		output = layers[i]->forward(output);
	}
	return output;
}

void ml::Model::train(const ml::Data& trainingData, const ml::Data& labels, int epoches) {
	if (lossFuncType == "cce") {
		if (dynamic_cast<ml::Activation*>(layers.back()) != nullptr) {
			ml::Layer* tmp = layers.back();
			layers.pop_back();
			this->train(trainingData, labels, epoches);
			layers.push_back(tmp);
			return;
		}
	}

	ml::Optimizer* opt = ml::Optimizers.at(optimizerType)(this, &layers);
	for (int i = 0; i < epoches; ++i) {
		opt->train(trainingData, labels);
	}

}

void ml::Model::applyRegularization(double lambda) {
	applyL2Regularization(lambda);
}

void ml::Model::applyL2Regularization(double lambda) {
	for (auto layer : layers) {
		layer->setParameters(layer->getParameters() * (1 - lambda));
	}
}