#ifndef MLL_FUNCTIONS_LOSS_FUNCTIONS_H_
#define MLL_FUNCTIONS_LOSS_FUNCTIONS_H_

#include "MLL/Linear/Tensor.h"
#include <functional>

namespace ml {
	// Mean Square Error
	double MSE(const ml::Tensor& predicted, const ml::Tensor& desired);
	ml::Tensor MSE_deriv(const ml::Tensor& predicted, const ml::Tensor& desired);

	// Binary Cross-Entropy
	double BCE(const ml::Tensor& predicted, const ml::Tensor& desired);
	ml::Tensor BCE_deriv(const ml::Tensor& predicted, const ml::Tensor& desired);

	// Categorical Cross-Entropy
	double CCE(const ml::Tensor& predicted, const ml::Tensor& desired);
	ml::Tensor CCE_deriv(const ml::Tensor& predicted, const ml::Tensor& desired);
}

namespace ml {
	extern const std::unordered_map<std::string, std::function<double(const ml::Tensor&, const ml::Tensor&)>> LossFunctions;
	extern const std::unordered_map<std::string, std::function<ml::Tensor(const ml::Tensor&, const ml::Tensor&)>> LossFuncDerivs;
}

#endif