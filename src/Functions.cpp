#include "MLL/Functions.h"
#include <stdexcept>
#include <cmath>
#include <limits>

// ----------------- ACTIVATION FUNCTIONS ----------------- //

#include <iostream>

ml::Tensor ml::ReLU(const ml::Tensor& tensor) {
	return tensor.applyFunc([](double v) -> double { return std::fmax(0.0f, v); });
}

ml::Tensor ml::ReLU_deriv(const ml::Tensor& tensor) {
	return tensor.applyFunc([](double v) -> double { return (v > 0.0f); });
}

ml::Tensor ml::leakyReLU(const ml::Tensor& tensor) {
	return tensor.applyFunc([](double v) -> double { return std::fmax(0.001f * v, v); });
}

ml::Tensor ml::leakyReLU_deriv(const ml::Tensor& tensor) {
	return tensor.applyFunc([](double v) -> double { return (v > 0.0f) ? 1.0f : 0.001f; });
}

ml::Tensor ml::sigmoid(const ml::Tensor& tensor) {
	return tensor.applyFunc([](double v) -> double { return v / 2.0f / (1.0f + std::fabs(v)) + 0.5f; });
}

ml::Tensor ml::sigmoid_deriv(const ml::Tensor& tensor) {
	return tensor.applyFunc([](double v) -> double {
		double t = 1.0f + std::fabs(v);
		return 0.5f / (t * t);
	});
}

ml::Tensor ml::softmax(const ml::Tensor& tensor) {
	ml::Tensor tmp = ml::exp(tensor);
	tmp /= tmp.sum();
	return tmp;
}

ml::Tensor ml::softmax_deriv(const ml::Tensor& tensor) {
	ml::Tensor tmp = ml::softmax(tensor);
	tmp *= (-tmp + 1);
	return tmp;
}

namespace ml {
	const std::unordered_map<std::string, std::function<ml::Tensor(const ml::Tensor&)>> ActivFunctions = {
		{ "relu", ml::ReLU },
		{ "leaky relu", ml::leakyReLU },
		{ "sigmoid", ml::sigmoid },
		{ "softmax", ml::softmax },
	};

	const std::unordered_map<std::string, std::function<ml::Tensor(const ml::Tensor&)>> ActivFuncDerivs = {
		{ "relu", ml::ReLU_deriv },
		{ "leaky relu", ml::leakyReLU_deriv },
		{ "sigmoid", ml::sigmoid_deriv },
		{ "softmax", ml::softmax_deriv },
	};
}

// ----------------- LOSS FUNCTIONS ----------------- //

double ml::MSE(const ml::Tensor& predicted, const ml::Tensor& desired) {
	return (ml::square(predicted - desired)).sum();
}

ml::Tensor ml::MSE_deriv(const ml::Tensor& predicted, const ml::Tensor& desired) {
	return (predicted - desired) * 2;
}

double ml::BCE(const ml::Tensor& predicted, const ml::Tensor& desired) {
	return -((desired * ml::log(predicted) + (-desired + 1) * ml::log(-predicted + 1)).sum());
}

ml::Tensor ml::BCE_deriv(const ml::Tensor& predicted, const ml::Tensor& desired) {
	return (predicted - desired) / (predicted * (-predicted + 1));
}

double ml::CCE(const ml::Tensor& predicted, const ml::Tensor& desired) {
	return -((desired * ml::log(predicted)).sum());
}

ml::Tensor ml::CCE_deriv(const ml::Tensor& predicted, const ml::Tensor& desired) {
	return predicted - desired;
}

namespace ml {
	const std::unordered_map<std::string, std::function<double(const ml::Tensor&, const ml::Tensor&)>> LossFunctions = {
		{ "mse", ml::MSE },
		{ "bce", ml::BCE },
		{ "cce", ml::CCE },
	};

	const std::unordered_map<std::string, std::function<ml::Tensor(const ml::Tensor&, const ml::Tensor&)>> LossFuncDerivs = {
		{ "mse", ml::MSE_deriv },
		{ "bce", ml::BCE_deriv },
		{ "cce", ml::CCE_deriv },
	};
}

// ----------------- OTHER FUNCTIONS ----------------- //

ml::Tensor ml::abs(const ml::Tensor& tensor) {
	auto tmp = tensor;
	return tmp.applyFunc([](double x) -> double { return std::abs(x); });
}

ml::Tensor ml::square(const ml::Tensor& tensor) {
	auto tmp = tensor;
	return tmp.applyFunc([](double x) -> double { return x * x; });
}

ml::Tensor ml::log(const ml::Tensor& tensor) {
	auto tmp = tensor;
	return tmp.applyFunc([](double x) -> double { return std::log(x); });
}

ml::Tensor ml::exp(const ml::Tensor& tensor) {
	auto tmp = tensor;
	return tmp.applyFunc([](double x) -> double {
		return std::exp(x);
	});
}

#include <iostream>
ml::Tensor ml::correlate2d(const ml::Tensor& input_, const ml::Tensor& kernel, const std::string& type) {
	if (input_.getShape().size() != 2 || kernel.getShape().size() != 2) {
		throw std::invalid_argument("Invalid arguments for getting block of a tensor.");
	}

	ml::Tensor input;
	if (type == "valid") {
		input = input_;
	} else if (type == "full") {
		input = input.reshape({
			input_.getShape()[0] + 2 * (kernel.getShape()[0] - 1),
			input_.getShape()[1] + 2 * (kernel.getShape()[1] - 1)
		});
		input.setBlock({kernel.getShape()[0] - 1, kernel.getShape()[1] - 1}, input_);
	}
	
	ml::Tensor res({input.getShape()[0] - kernel.getShape()[0] + 1, input.getShape()[1] - kernel.getShape()[1] + 1});
	for (int i = 0; i < res.getShape()[0]; ++i) {
		for (int j = 0; j < res.getShape()[1]; ++j) {
			res({i, j}) = (input.block({i, j}, kernel.getShape()) * kernel).sum();
		}
	}

	return res;
}

ml::Tensor ml::convolve2d(const ml::Tensor& input, const ml::Tensor& kernel, const std::string& type) {
	return ml::correlate2d(input, kernel.reverse(), type);
}

std::vector<double> ml::strToDoubleVec(const std::string& input) {
	std::vector<double> data;
	std::string num;

	for (char c : input) {
		if (std::isspace(c)) {
			if (!num.empty()) {
				data.push_back(std::stod(num));
				num.clear();
			}
		} else {
			num += c;
		}
	}

	if (!num.empty()) {
		data.push_back(std::stod(num));
	}

	return data;
}

std::vector<int> ml::strToIntVec(const std::string& input) {
	std::vector<int> data;
	std::string num;

	for (char c : input) {
		if (std::isspace(c)) {
			if (!num.empty()) {
				data.push_back(std::stoi(num));
				num.clear();
			}
		} else {
			num += c;
		}
	}

	if (!num.empty()) {
		data.push_back(std::stoi(num));
	}

	return data;
}