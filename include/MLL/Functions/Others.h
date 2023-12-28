#ifndef MLL_FUNCTIONS_OTHERS_H_
#define MLL_FUNCTIONS_OTHERS_H_

#include "MLL/Linear/Tensor.h"
#include <vector>
#include <string>

namespace ml {
	ml::Tensor abs(const ml::Tensor& tensor);
	ml::Tensor square(const ml::Tensor& tensor);
	ml::Tensor log(const ml::Tensor& tensor);
	ml::Tensor exp(const ml::Tensor& tensor);

	ml::Tensor correlate2d(const ml::Tensor& input, const ml::Tensor& kenrel, const std::string& type);
	ml::Tensor convolve2d(const ml::Tensor& input, const ml::Tensor& kenrel, const std::string& type);

	std::vector<double> strToDoubleVec(const std::string& input);
	std::vector<int> strToIntVec(const std::string& input);
}

#endif