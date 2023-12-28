#ifndef MLL_FUNCTIONS_ACTIV_FUNCTIONS_H_
#define MLL_FUNCTIONS_ACTIV_FUNCTIONS_H_

#include "MLL/Linear/Tensor.h"
#include <unordered_map>
#include <functional>
#include <string>

namespace ml {
	ml::Tensor ReLU(const ml::Tensor& tensor);
	ml::Tensor ReLU_deriv(const ml::Tensor& tensor);

	ml::Tensor leakyReLU(const ml::Tensor& tensor);
	ml::Tensor leakyReLU_deriv(const ml::Tensor& tensor);

	ml::Tensor sigmoid(const ml::Tensor& tensor);
	ml::Tensor sigmoid_deriv(const ml::Tensor& tensor);

	ml::Tensor softmax(const ml::Tensor& tensor);
	ml::Tensor softmax_deriv(const ml::Tensor& tensor);
}

namespace ml {
	extern const std::unordered_map<std::string, std::function<ml::Tensor(const ml::Tensor&)>> ActivFunctions;
	extern const std::unordered_map<std::string, std::function<ml::Tensor(const ml::Tensor&)>> ActivFuncDerivs;
}

#endif